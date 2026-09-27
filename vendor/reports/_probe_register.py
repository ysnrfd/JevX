"""Probe 2: Q3 registration flow, Q6 eval-mode equality, Q8 processor call."""
import sys, os, json, tempfile
sys.path.insert(0, r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src')
import torch
from transformers import (BertConfig, BertModel, T5Config, T5Model,
                          PreTrainedModel, PreTrainedConfig,
                          AutoConfig, AutoModel, AutoTokenizer, AutoProcessor)

results = {}

# ---- Q3: registration mechanisms ----
class RegConfig(PreTrainedConfig):
    model_type = "jevx_reg"
    def __init__(self, hidden_size=32, **kwargs):
        self.hidden_size = hidden_size
        self.backbone = kwargs.pop("backbone", None)
        super().__init__(**kwargs)

class RegModel(PreTrainedModel):
    config_class = RegConfig
    base_model_prefix = "jevx"
    def __init__(self, config):
        super().__init__(config)
        bb = BertConfig.from_dict(config.backbone) if config.backbone else BertConfig(hidden_size=config.hidden_size)
        self.backbone = BertModel(bb)
        self.head = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, **kw):
        o = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return type("O", (), {"reps": self.head(o.last_hidden_state[:, 0])})()

# (a) AutoConfig.register + AutoModel.register
try:
    AutoConfig.register("jevx_reg", RegConfig, exist_ok=True)
    AutoModel.register(RegConfig, RegModel, exist_ok=True)
    results['q3_register_ok'] = True
except Exception as e:
    results['q3_register_ok'] = f"{type(e).__name__}: {e}"

cfg = RegConfig()
cfg.backbone = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                          intermediate_size=64, vocab_size=100).to_dict()
m = RegModel(cfg)
ids = torch.randint(0, 100, (2, 7))
m(input_ids=ids)
with tempfile.TemporaryDirectory() as d:
    m.save_pretrained(d)
    try:
        ac = AutoConfig.from_pretrained(d)
        results['q3_autoconfig_after_register'] = type(ac).__name__
    except Exception as e:
        results['q3_autoconfig_after_register'] = f"{type(e).__name__}: {str(e)[:200]}"
    try:
        am = AutoModel.from_pretrained(d)
        results['q3_automodel_after_register'] = type(am).__name__
    except Exception as e:
        results['q3_automodel_after_register'] = f"{type(e).__name__}: {str(e)[:200]}"

# (b) register_for_auto_class -> auto_map written on save
class RemoteConfig(PreTrainedConfig):
    model_type = "jevx_remote"
    def __init__(self, hidden_size=32, **kwargs):
        self.hidden_size = hidden_size
        self.backbone = kwargs.pop("backbone", None)
        super().__init__(**kwargs)
    @classmethod
    def register_for_auto_class(cls, auto_class="AutoConfig"):
        return super().register_for_auto_class(auto_class)

class RemoteModel(PreTrainedModel):
    config_class = RemoteConfig
    base_model_prefix = "jevx"
    def __init__(self, config):
        super().__init__(config)
        bb = BertConfig.from_dict(config.backbone) if config.backbone else BertConfig(hidden_size=config.hidden_size)
        self.backbone = BertModel(bb)
        self.head = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.post_init()
    def forward(self, input_ids=None, attention_mask=None, **kw):
        o = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return type("O", (), {"reps": self.head(o.last_hidden_state[:, 0])})()

RemoteConfig.register_for_auto_class("AutoConfig")
RemoteModel.register_for_auto_class("AutoModel")
rcfg = RemoteConfig()
rcfg.backbone = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
                           intermediate_size=64, vocab_size=100).to_dict()
rm = RemoteModel(rcfg)
rm(input_ids=ids)
with tempfile.TemporaryDirectory() as d:
    rm.save_pretrained(d)
    cj = json.load(open(os.path.join(d, 'config.json')))
    results['q3_auto_map_written'] = cj.get('auto_map')
    results['q3_source_files_copied'] = sorted(f for f in os.listdir(d) if f.endswith('.py'))

# ---- Q6: T5 encoder equivalence under eval (dropout confound) ----
t5c = T5Config(vocab_size=100, d_model=32, d_ff=64, num_layers=2, num_heads=4, num_decoder_layers=1)
t5 = T5Model(t5c)
t5.eval()
dec = torch.randint(0, 100, (2, 5))
with torch.no_grad():
    full = t5(input_ids=ids, decoder_input_ids=dec, output_hidden_states=True)
    enc = t5.encoder(input_ids=ids)
results['q6_eval_encoder_equals_field'] = bool(torch.equal(enc.last_hidden_state, full.encoder_last_hidden_state))
results['q6_encoder_hidden_states_count'] = None if full.encoder_hidden_states is None else len(full.encoder_hidden_states)
# also call via t5.get_encoder() if exists
results['q6_has_get_encoder'] = hasattr(t5, 'get_encoder')

# ---- Q8: processor resolution on a known multimodal repo (config-level only, no download of weights) ----
# Just check class availability + call signature shape using a local tiny CLIP processor? Skip network.
from transformers import CLIPProcessor, SiglipProcessor, WhisperProcessor, Wav2Vec2Processor
results['q8_processor_classes_importable'] = True
import inspect
sig = inspect.signature(CLIPProcessor.__call__)
results['q8_clip_processor_call_params'] = list(sig.parameters.keys())[:8]
from transformers.processing_utils import ProcessorMixin
sig2 = inspect.signature(ProcessorMixin.__call__)
results['q8_processor_mixin_call_params'] = list(sig2.parameters.keys())

# AutoProcessor resolution internals: PROCESSOR_MAPPING exists
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING, AutoProcessor
results['q8_processor_mapping_is_lazy'] = type(PROCESSOR_MAPPING).__name__

print(json.dumps(results, indent=2, default=str))
