"""Build vendor/reports/architecture_inventory.json from transformers v5.16.1 Auto mappings.

Programmatically iterates CONFIG_MAPPING and all MODEL_FOR_* / MODEL_* mappings and records,
per model_type: config class, base model class, which AutoModelFor* heads exist, plus notes
cheaply derivable from the source (is_encoder_decoder, model_type value).
Throwaway script; output is the deliverable JSON.
"""
import sys, os, json, re, collections
sys.path.insert(0, r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\transformers\src')

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
import transformers.models.auto.modeling_auto as ma

OUT = r'C:\Users\ysnrfd\Documents\Transformer\JevX\vendor\reports\architecture_inventory.json'

# All MODEL_*_MAPPING_NAMES style OrderedDicts in modeling_auto
# (exclude CONFIG_MAPPING_NAMES — it is re-exported there but is a config mapping, not a task mapping)
mapping_names = {
    name: obj for name, obj in vars(ma).items()
    if name.startswith('MODEL_') and name.endswith('_MAPPING_NAMES') and isinstance(obj, collections.OrderedDict)
}

records = {}
for model_type, config_cls_name in CONFIG_MAPPING.items():
    records[model_type] = {
        'model_type': model_type,
        'config_class': config_cls_name,
        'base_model_class': None,
        'auto_for_tasks': {},
    }

for mapping_name, od in mapping_names.items():
    if mapping_name == 'MODEL_MAPPING_NAMES':
        continue
    task = mapping_name.replace('MODEL_FOR_', '').replace('_MAPPING_NAMES', '')
    for model_type, cls_name in od.items():
        rec = records.setdefault(model_type, {
            'model_type': model_type, 'config_class': None,
            'base_model_class': None, 'auto_for_tasks': {}})
        rec['auto_for_tasks'][task] = cls_name

for model_type, cls_name in mapping_names.get('MODEL_MAPPING_NAMES', {}).items():
    rec = records.setdefault(model_type, {
        'model_type': model_type, 'config_class': None,
        'base_model_class': None, 'auto_for_tasks': {}})
    rec['base_model_class'] = cls_name

# Try to enrich with config metadata (is_encoder_decoder, architecture module) — best effort
from transformers import AutoConfig  # noqa: F401
enriched = 0
for model_type, rec in records.items():
    cfg_name = rec['config_class']
    if not cfg_name:
        continue
    # find module via CONFIG_MAPPING value already is class name; get class
    try:
        cfg_cls = CONFIG_MAPPING[model_type]
        rec['config_class'] = cfg_cls.__name__
        rec['config_module'] = cfg_cls.__module__
        rec['is_encoder_decoder'] = bool(getattr(cfg_cls, 'is_encoder_decoder', False))
        rec['config_model_type_attr'] = getattr(cfg_cls, 'model_type', model_type)
        enriched += 1
    except Exception as e:
        rec['enrich_error'] = f'{type(e).__name__}: {e}'

# Resolve base model class module (where defined) for base models
for model_type, rec in records.items():
    bn = rec.get('base_model_class')
    if not bn:
        continue
    # e.g. 'BertModel' -> transformers.models.bert.modeling_bert
    # derive from config module replacing configuration_ with modeling_
    cm = rec.get('config_module')
    if cm:
        rec['base_model_module'] = cm.replace('configuration_', 'modeling_')

out = {
    'transformers_version': __import__('transformers').__version__,
    'generated_from': 'vendor/transformers/src/transformers/models/auto (CONFIG_MAPPING + *_MAPPING_NAMES)',
    'record_count': len(records),
    'mappings_seen': sorted(mapping_names.keys()),
    'records': sorted(records.values(), key=lambda r: r['model_type']),
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('records:', out['record_count'], 'enriched:', enriched)
print('mappings:', len(out['mappings_seen']))
