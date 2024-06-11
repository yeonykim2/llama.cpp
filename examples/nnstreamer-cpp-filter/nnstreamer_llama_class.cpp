#include "common.h"
#include "console.h"
#include "llama.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <signal.h>
#include <unistd.h>

#include <tensor_filter_cpp.hh>


class nnstreamer_llama_filter : public tensor_filter_cpp
{
  private:
  struct llama_params {

    int32_t n_predict = 128;
    std::string model     = "models/llama-68m-chat-v1.fp16.gguf";

    std::string s_prompt = "<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Please, summarize the sutitle.\n";
    std::string e_prompt = "<|im_end|>";

    bool input_echo           = true;
    bool display              = true;

    int n_past             = 0;
    int n_remain           = 0;
    int n_consumed         = 0;
    int n_session_consumed = 0;
    int n_past_guidance    = 0;

  };

  llama_params lparams;
  gpt_params params;
  llama_sampling_params sparams;
  llama_model * model;
  llama_context * ctx;
  llama_context * ctx_guidance = NULL;

  int n_ctx_train;
  int n_ctx;
  bool add_bos;

  std::vector<llama_token> embd_inp;

  std::vector<int>   input_tokens;
  std::vector<int>   output_tokens;
  std::ostringstream output_ss;
  std::vector<llama_token> embd;
  std::vector<llama_token> embd_guidance;
  
  struct llama_sampling_context * ctx_sampling;

  nnstreamer_llama_filter (const char *modelName) : tensor_filter_cpp (modelName)
  {
    params.model = lparams.model;
    params.n_predict = lparams.n_predict;
    sparams = params.sparams;
    
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);
    LOG_TEE("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_TEE("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n", __func__);
    }

    n_ctx_train = llama_n_ctx_train(model);
    n_ctx = llama_n_ctx(ctx);


    LOG_TEE("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", gpt_params_get_system_info(params).c_str());
    }

    add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG_TEE("add_bos: %d\n", add_bos);
 
  }

  ~nnstreamer_llama_filter ()
  {
    llama_print_timings(ctx);
    llama_free(ctx);
    llama_free_model(model);
    llama_sampling_free(ctx_sampling);
    llama_backend_free();
  }

  public:
  int getInputDim (GstTensorsInfo *info)
  {
    info->num_tensors = 1;
    info->info[0].type = _NNS_UINT8;
    info->info[0].dimension[0] = 48000;
    info->info[0].dimension[1] = 1;
    info->info[0].dimension[2] = 1;
    for (int i = 3; i < NNS_TENSOR_RANK_LIMIT; i++)
      info->info[0].dimension[i] = 1;

    return 0;
  }

  int getOutputDim (GstTensorsInfo *info)
  {
    info->num_tensors = 1;
    info->info[0].type = _NNS_UINT8;
    info->info[0].dimension[0] = 48000;
    info->info[0].dimension[1] = 1;
    info->info[0].dimension[2] = 1;
    info->info[0].dimension[3] = 1;
    for (int i = 4; i < NNS_TENSOR_RANK_LIMIT; i++)
      info->info[0].dimension[i] = 1;

    return 0;
  }

  int setInputDim (const GstTensorsInfo *in, GstTensorsInfo *out)
  {
    return -EINVAL;
  }

  bool isAllocatedBeforeInvoke ()
  {
    return true;
  }

  int invoke (const GstTensorMemory *in, GstTensorMemory *out)
  {
    params.prompt = lparams.s_prompt + (char*)in->data + lparams.e_prompt;
    
    embd_inp = ::llama_tokenize(ctx, params.prompt, true, true);
    LOG_TEE("prompt: *******\n \"%s\"\n ******* \n", log_tostr(params.prompt));

    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG_TEE("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }
    LOGLN(
            "recalculate the cached logits (check): embd_inp.empty() %s,  embd_inp.size() %zu, embd_inp.size() %zu",
            log_tostr(embd_inp.empty()), embd_inp.size(), embd_inp.size());

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size() || params.instruct || params.chatml) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_TEE("\n\n");

    console::set_display(console::prompt);
    lparams.display = params.display_prompt;

    lparams.n_remain = params.n_predict;
    ctx_sampling = llama_sampling_init(sparams);
    if (!ctx_sampling) {
        fprintf(stderr, "%s: failed to initialize sampling subsystem\n", __func__);
        exit(1);
    }

    while (lparams.n_remain != 0) {
      // predict
      if (!embd.empty()) {
          // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
          // --prompt or --file which uses the same value.
          int max_embd_size = n_ctx - 4;

          // Ensure the input doesn't exceed the context size by truncating embd if necessary.
          if ((int) embd.size() > max_embd_size) {
              const int skipped_tokens = (int) embd.size() - max_embd_size;
              embd.resize(max_embd_size);

              console::set_display(console::error);
              printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
              console::set_display(console::reset);
              fflush(stdout);
          }

          for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
              int n_eval = (int) embd.size() - i;
              if (n_eval > params.n_batch) {
                  n_eval = params.n_batch;
              }

              LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

              if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, lparams.n_past, 0))) {
                  LOG_TEE("%s : failed to eval\n", __func__);
              }

              lparams.n_past += n_eval;

              LOG("n_past = %d\n", lparams.n_past);
              // Display total tokens alongside total time
              if (params.n_print > 0 && lparams.n_past % params.n_print == 0) {
                  LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", lparams.n_past, n_ctx);
              }
          }

      }
      embd.clear();
      embd_guidance.clear();
      if ((int) embd_inp.size() <= lparams.n_consumed) {

        const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);

        llama_sampling_accept(ctx_sampling, ctx, id, /* apply_grammar= */ true);

        LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

        embd.push_back(id);

        // echo this to console
        lparams.input_echo = true;

        // decrement remaining sampling budget
        --lparams.n_remain;

        LOG("n_remain: %d\n", lparams.n_remain);
      } else {
          // some user input remains from prompt or interaction, forward it to processing
          LOG_TEE("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), lparams.n_consumed);
          while ((int) embd_inp.size() > lparams.n_consumed) {
              embd.push_back(embd_inp[lparams.n_consumed]);

              // push the prompt in the sampling context in order to apply repetition penalties later
              // for the prompt, we don't apply grammar rules
              llama_sampling_accept(ctx_sampling, ctx, embd_inp[lparams.n_consumed], /* apply_grammar= */ false);

              ++lparams.n_consumed;
              if ((int) embd.size() >= params.n_batch) {
                  break;
              }
          }
      }

      // display text
      if (lparams.input_echo && lparams.display) {
          for (auto id : embd) {
              const std::string token_str = llama_token_to_piece(ctx, id, params.special);

              // Console/Stream Output
              fprintf(stdout, "%s", token_str.c_str());

              // Record Displayed Tokens To Log
              // Note: Generated tokens are created one by one hence this check
              if (embd.size() > 1) {
                  // Incoming Requested Tokens
                  input_tokens.push_back(id);
              } else {
                  // Outgoing Generated Tokens
                  output_tokens.push_back(id);
                  output_ss << token_str;
              }

              fflush(stdout);
          }
      }
      // reset color to default if there is no pending user input
      if (lparams.input_echo && (int) embd_inp.size() == lparams.n_consumed) {
          console::set_display(console::reset);
          lparams.display = true;
      }

      // if not currently processing queued inputs;
      if ((int) embd_inp.size() <= lparams.n_consumed) {
          // deal with end of generation tokens in interactive mode
          if (llama_token_is_eog(model, llama_sampling_last(ctx_sampling))) {
              LOG_TEE("\nfound an EOG token\n");
          }
      }

      // end of generation
      if (!embd.empty() && llama_token_is_eog(model, embd.back()) && !(params.instruct || params.interactive || params.chatml)) {
          LOG_TEE(" [end of text]\n");
          break;
      }

    }

    return 0;
  }

  static nnstreamer_llama_filter &get_instance ()
  {
    static nnstreamer_llama_filter instance ("nnstreamer_llama_filter");
    return instance;
  }
};

void init_shared_lib (void) __attribute__ ((constructor));

void
init_shared_lib (void)
{
  nnstreamer_llama_filter &mccf = nnstreamer_llama_filter::get_instance ();
  mccf._register ();
}
