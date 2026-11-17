"""Recompute all 360 recorded profiles on CPU; never submits device work."""
from pathlib import Path
import hashlib,json,statistics,sys,tarfile,tempfile
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'adapter'))
from hardware_timer import parse_hardware

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
archive=root/'raw-evidence.tar.gz'
assert sha(archive)==(root/'SHA256SUMS').read_text().split()[0]
prov=json.loads((root/'provenance.json').read_text())
for f,h in prov['measured_adapter_sha256'].items():assert sha(root/'adapter'/f)==h,f
with tempfile.TemporaryDirectory() as tmp:
 data=Path(tmp)
 with tarfile.open(archive) as tar:
  for m in tar.getmembers():
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   target=data/m.name;target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(tar.extractfile(m).read())
 for f,h in json.loads((data/'SHA256.json').read_text()).items():assert sha(data/f)==h,f
 for f,h in json.loads((data/'byte-identical-raw-sha256.json').read_text()).items():assert sha(data/f)==h,f
 guards=json.loads((data/'guard-receipts.json').read_text())
 for g in guards:
  assert g['phase']=='finished' and g['exit_code']==g['workload_exit_code']==0 and g['performance_valid']
  assert not g['conflict_detected'] and not g['visibility_lost']
 accuracy=json.loads((data/'results/moe-accuracy.json').read_text())
 assert len(accuracy['cases'])==12
 for c in accuracy['cases']:
  for side in ['tilelang','aclnn-v3-composed']:
   assert c[side]['accuracy']=='passed' and c[side]['outputs_checked']==3 and c[side]['input_preserved']
  assert c['tilelang']['nondefault_stream']=='passed'
 count=0;rows=[]
 for f in sorted((data/'results').glob('moe-performance-*.json')):
  d=json.loads(f.read_text());assert d['identity']['limits']=={'cube_core_num':32,'vector_core_num':64}
  for c in d['cases']:
   for side in ['tilelang','aclnn-v3-composed']:
    t=c[side]['timing'];assert len(t['samples'])==15
    for s in t['samples']:
     parsed=parse_hardware(data/s['raw_path'])
     for k in ['device_union_us','ai_core_union_us','aicpu_union_us','device_span_us','tasks']:
      assert json.loads(json.dumps(parsed[k]))==s[k],(c['case'],side,s['index'],k)
     assert s['frequency_before']==s['frequency_after']==[1650,1650]
     assert len(s['tasks'])==(1 if side=='tilelang' else 27)
     count+=1
    for k in ['device_union_us','ai_core_union_us','aicpu_union_us','device_span_us']:
     assert statistics.mean(s[k] for s in t['samples'])==t['mean_'+k]
   assert c['frequency_comparable']
   tl=c['tilelang']['timing']['mean_device_union_us'];base=c['aclnn-v3-composed']['timing']['mean_device_union_us']
   assert abs(c['ratio']-base/tl)<1e-12
   rows.append(dict(case=c['case'],tilelang_us=tl,baseline_us=base,baseline_aicpu_us=c['aclnn-v3-composed']['timing']['mean_aicpu_union_us'],ratio=c['ratio']))
 assert count==360 and len(rows)==12 and len(set(c['case'] for c in rows))==12
 print(json.dumps(dict(replayed_profiles=count,cases=len(rows),byte_identical_raw_files=1800,guard_receipts=len(guards),rows=rows),indent=2))
