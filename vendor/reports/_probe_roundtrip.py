"""Throwaway probe: verify Q1 (wrapper round-trip), Q2 (return_dict), Q4 (hidden_states), Q5, Q6, Q7."""
import sys, os, json, tempfile
sys.path.insert(0, r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src')
import torch
from transformers import BertConfig, BertModel, LlamaConfig, LlamaModel, AutoModel, AutoConfig
from transformers import PreTrainedModel, PreTrainedConfig

results = {}

# ---------- Q1: wrapper holding an HF model as child ----------
class TinyJevXConfig(PreTrainedConfig):
    model_type = "jevx_tiny"
    def __init__(self, hidden_size=32, **kwargs):
        self.hidden_size = hidden_size
        self.backbone = kwargs.pop("backbone", None)  # dict form of inner config
        super().__init__(**kwargs)

class TinyJevXModel(PreTrainedModel):
    config_class = TinyJevXConfig
    base_model_prefix = "jevx"
    def __init__(self, config):
        super().__init__(config)
        bb_config = BertConfig.from_dict(config.backbone) if config.backbone else BertConfig(hidden_size=config.hidden_size)
        self.backbone = BertModel(bb_config)
        self.head = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, **kw):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        pooled = out.last_hidden_state[:, 0]
        return type("Out", (), {"reps": self.head(pooled), "hidden_states": out.hidden_states})()

cfg = TinyJevXConfig()
cfg.backbone = BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100).to_dict()
m = TinyJevXModel(cfg)
ids = torch.randint(0, 100, (2, 7))
m(input_ids=ids)

with tempfile.TemporaryDirectory() as d:
    try:
        m.save_pretrained(d)
        files = sorted(os.listdir(d))
        results['q1_save_files'] = files
        m2 = TinyJevXModel.from_pretrained(d)
        sd1, sd2 = m.state_dict(), m2.state_dict()
        same = all(torch.equal(sd1[k], sd2[k]) for k in sd1)
        results['q1_roundtrip_weights_equal'] = same
        ac = AutoConfig.from_pretrained(d)
        results['q1_autoconfig_model_type'] = ac.model_type
        cj = json.load(open(os.path.join(d, 'config.json')))
        results['q1_config_architectures'] = cj.get('architectures')
        results['q1_config_has_backbone_model_type'] = cj.get('backbone', {}).get('model_type')
        results['q1_config_auto_map'] = cj.get('auto_map')
        am = AutoModel.from_pretrained(d)
        results['q1_automodel_class'] = type(am).__name__
    except Exception as e:
        results['q1_error'] = f"{type(e).__name__}: {e}"

# safetensors shared-tensor test: alias module not declared in tied keys
class TiedJevXConfig(PreTrainedConfig):
    model_type = "jevx_tied"

class TiedJevXModel(PreTrainedModel):
    config_class = TiedJevXConfig
    def __init__(self, config):
        super().__init__(config)
        self.b = BertModel(BertConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=50))
        self.alias = self.b.embeddings.word_embeddings  # same module referenced twice
        self.post_init()
    def forward(self, input_ids=None, **k):
        return self.b(input_ids=input_ids)

tc = TiedJevXConfig()
tm = TiedJevXModel(tc)
tm(input_ids=torch.randint(0, 50, (1, 4)))
with tempfile.TemporaryDirectory() as d:
    try:
        tm.save_pretrained(d)
        results['q1_illegal_alias_save'] = 'succeeded'
    except Exception as e:
        results['q1_illegal_alias_save'] = f"{type(e).__name__}: {str(e)[:300]}"

# ---------- Q2: return_dict=False ----------
bconf = BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100)
bm = BertModel(bconf)
out = bm(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=False)
results['q2_bert_return_dict_false_type'] = type(out).__name__
results['q2_bert_return_dict_false_is_tuple'] = isinstance(out, tuple)
results['q2_bert_return_dict_false_len'] = len(out) if isinstance(out, tuple) else None
out2 = bm(input_ids=ids, attention_mask=torch.ones_like(ids))
results['q2_bert_default_type'] = type(out2).__name__

lconf = LlamaConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, intermediate_size=64, vocab_size=100, use_cache=False)
lm = LlamaModel(lconf)
lo = lm(input_ids=ids, return_dict=False)
results['q2_llama_return_dict_false_type'] = type(lo).__name__
results['q2_llama_return_dict_false_is_tuple'] = isinstance(lo, tuple)
lo2 = lm(input_ids=ids)
results['q2_llama_default_type'] = type(lo2).__name__

bconf2 = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=100, return_dict=False)
bm2 = BertModel(bconf2)
o = bm2(input_ids=ids, attention_mask=torch.ones_like(ids))
results['q2_config_return_dict_false_type'] = type(o).__name__

# ---------- Q4: output_hidden_states ----------
bconf3 = BertConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64, vocab_size=100)
bm3 = BertModel(bconf3)
o = bm3(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
results['q4_hidden_states_len'] = len(o.hidden_states)
results['q4_hidden_states_shape0'] = tuple(o.hidden_states[0].shape)
results['q4_last_equals_hidden_states_last'] = bool(torch.equal(o.last_hidden_state, o.hidden_states[-1]))
results['q4_base_fields_present'] = [k for k in o.keys()]

bconf3.output_hidden_states = True
bm4 = BertModel(bconf3)
o4 = bm4(input_ids=ids, attention_mask=torch.ones_like(ids))
results['q4_config_flag_works'] = o4.hidden_states is not None
o5 = bm4(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=False)
results['q4_toggle_off_gives_none'] = o5.hidden_states is None
lo3 = lm(input_ids=ids, output_hidden_states=True)
results['q4_llama_hidden_states_len'] = len(lo3.hidden_states) if lo3.hidden_states else None
results['q4_llama_last_tied'] = bool(torch.equal(lo3.last_hidden_state, lo3.hidden_states[-1])) if lo3.hidden_states else None

# ---------- Q5: dtype conventions ----------
lc = LlamaConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32, vocab_size=50)
lm5 = LlamaModel(lc)
with tempfile.TemporaryDirectory() as d:
    lm5.save_pretrained(d)
    cj = json.load(open(os.path.join(d, 'config.json')))
    results['q5_saved_config_dtype'] = cj.get('dtype')
    results['q5_saved_config_has_torch_dtype_key'] = 'torch_dtype' in cj
    m_bf = LlamaModel.from_pretrained(d, dtype=torch.bfloat16)
    results['q5_dtype_kwarg_works'] = str(next(m_bf.parameters()).dtype)
    try:
        m_td = LlamaModel.from_pretrained(d, torch_dtype=torch.bfloat16)
        results['q5_torch_dtype_kwarg_works'] = str(next(m_td.parameters()).dtype)
    except Exception as e:
        results['q5_torch_dtype_kwarg_works'] = f"error: {e}"
    m_auto = LlamaModel.from_pretrained(d, dtype="auto")
    results['q5_dtype_auto'] = str(next(m_auto.parameters()).dtype)
    wd = TinyJevXConfig()
    wd.backbone = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, vocab_size=100).to_dict()
    wm = TinyJevXModel(wd)
    with tempfile.TemporaryDirectory() as d2:
        wm.save_pretrained(d2)
        try:
            wm2 = TinyJevXModel.from_pretrained(d2, dtype=torch.bfloat16)
            results['q5_wrapper_dtype'] = str(next(wm2.parameters()).dtype)
            results['q5_wrapper_child_dtype'] = str(next(wm2.backbone.parameters()).dtype)
        except Exception as e:
            results['q5_wrapper_dtype'] = f"error: {e}"

# ---------- Q6: t5 encoder ----------
from transformers import T5Config, T5Model
t5c = T5Config(vocab_size=100, d_model=32, d_ff=64, num_layers=1, num_heads=4, num_decoder_layers=1)
t5 = T5Model(t5c)
dec_ids = torch.randint(0, 100, (2, 5))
so = t5(input_ids=ids, decoder_input_ids=dec_ids, output_hidden_states=True)
results['q6_t5_output_type'] = type(so).__name__
results['q6_t5_encoder_last_hidden_state_shape'] = tuple(so.encoder_last_hidden_state.shape)
results['q6_t5_encoder_hidden_states'] = None if so.encoder_hidden_states is None else len(so.encoder_hidden_states)
enc_out = t5.encoder(input_ids=ids)
results['q6_t5_direct_encoder_type'] = type(enc_out).__name__
results['q6_t5_direct_encoder_has_lhn'] = hasattr(enc_out, 'last_hidden_state')
enc_out2 = t5.encoder(input_ids=ids, return_dict=False)
results['q6_t5_encoder_return_dict_false_type'] = type(enc_out2).__name__
results['q6_encoder_direct_equals_field'] = bool(torch.equal(enc_out.last_hidden_state, so.encoder_last_hidden_state))

# ---------- Q7: pooler presence (tiny configs to avoid OOM) ----------
from transformers import RobertaModel, DistilBertModel, AlbertModel, DebertaV2Model, GPT2Model, Qwen2Model, MistralModel, GemmaModel, Phi3Model

def tiny_cfg(cls, **over):
    c = cls()
    for k, v in dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                     num_key_value_heads=2, intermediate_size=64, vocab_size=100,
                     embed_dim=32, dim=32, hidden_dim=64, n_embd=32, n_layer=1, n_head=4,
                     d_model=32, d_ff=64, max_position_embeddings=64).items():
        if hasattr(c, k):
            try:
                setattr(c, k, v)
            except Exception:
                pass
    for k, v in over.items():
        setattr(c, k, v)
    return c

has_pooler = {}
for name, cls in [('bert', BertModel), ('roberta', RobertaModel), ('distilbert', DistilBertModel), ('albert', AlbertModel)]:
    try:
        inst = cls(tiny_cfg(cls.config_class))
        has_pooler[name] = hasattr(inst, 'pooler') and inst.pooler is not None
    except Exception as e:
        has_pooler[name] = f"err {type(e).__name__}: {e}"
for name, cls in [('llama', LlamaModel), ('gpt2', GPT2Model), ('qwen2', Qwen2Model), ('mistral', MistralModel), ('gemma', GemmaModel), ('phi3', Phi3Model)]:
    try:
        inst = cls(tiny_cfg(cls.config_class))
        has_pooler[name] = hasattr(inst, 'pooler') and inst.pooler is not None
    except Exception as e:
        has_pooler[name] = f"err {type(e).__name__}: {e}"
try:
    dv = DebertaV2Model(tiny_cfg(DebertaV2Model.config_class, rel_attn=False))
    has_pooler['deberta_v2_base_model_has_pooler_attr'] = hasattr(dv, 'pooler')
except Exception as e:
    has_pooler['deberta_v2'] = f"err {type(e).__name__}: {e}"
results['q7_pooler_presence'] = has_pooler

import transformers, importlib.util
results['q7_has_transformers_pooling_export'] = [n for n in dir(transformers) if 'ool' in n and not n.startswith('_')][:20]
results['q7_sentence_transformers_installed'] = importlib.util.find_spec('sentence_transformers') is not None

print(json.dumps(results, indent=2, default=str))
