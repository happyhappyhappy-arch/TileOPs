# PR44 complete ACLNN V3 baseline evidence

This evidence-only branch accompanies [TileOPs PR44](https://github.com/yyttt6/TileOPs/pull/44). It does not change the PR kernel or its CI baseline selection. The measurements qualify the stated self-test comparison, not the separate formal CANN 9.2 environment.

## Audit entry points

- [Complete adapter](adapter/baseline.py): measured callable `full(ids, num_experts, h=0)`. All routing and output adaptation dependent on input data execute on the NPU.
- [Measured runner](adapter/run_moe.py): the exact frozen runner, including seed 1235, manifest enumeration, complete three-output validator, input preservation and nondefault-stream checks. It is a historical source snapshot, with installation identity assertions and a reservation requirement; adapt these to the formal runner before future use. Do not execute it outside the device queue.
- [Timing implementation](adapter/hardware_timer.py) and [profiler capture configuration](adapter/capture_support.py).
- [Raw evidence archive](raw-evidence.tar.gz), [archive checksum](SHA256SUMS) and [source/environment provenance](provenance.json).
- [CPU replay](replay.py): validates archive/file/adapter hashes and reconstructs all 360 profiles, all task intervals and all 12 case means. It does not import torch or initialize an NPU.

On Linux with Python 3.10 or later, run:

```sh
python replay.py
```

The archive contains `results/moe-accuracy.json`, three `results/moe-performance-*.json` files, guard receipt extracts, and 360 per-call profile directories. `profiles/<case>/<implementation>/sample-<NN>/` retains FFTS, STARS, task-track and hash-dictionary bytes. These 1,800 raw files are byte-identical to the returned measurements. Device metadata retains the original clock conversion and device/version fields, while unrelated host identity metadata is omitted. Installation paths are replaced and raw paths made relative in result JSON; numeric measurements and task lists are unchanged. Full redaction notes and original archive/receipt hashes are provided. This is a minimal raw-record replay bundle, not a complete msprof application trace.

The measured manifest, workload generator and case binding match the published candidate: [manifest](https://github.com/happyhappyhappy-arch/TileOPs/blob/b691bc4f103d81fb5ab47d7ca42618232736b5dd/src/tileops/manifest/spec/moe.yaml), [workloads](https://github.com/happyhappyhappy-arch/TileOPs/blob/b691bc4f103d81fb5ab47d7ca42618232736b5dd/workloads/moe.py), [case binding](https://github.com/happyhappyhappy-arch/TileOPs/blob/b691bc4f103d81fb5ab47d7ca42618232736b5dd/benchmarks/_cases/moe.py).

## Full workload boundary

`aclnnMoeInitRoutingV3` alone does not produce this operator's aligned three-output contract. The measured adapter uses dropless routing with N=T*K, H=0, rowIdxType=1 and expert counts. H=0 omits irrelevant activation copying; its support was validated only in the stated CANN 9.3 environment.

The complete measured callable then computes counts/prefix sums, 64-route aligned expert starts, route destination indices and scatter, padded route IDs, block owners (including SearchSorted), and the final padded count. It returns all three int32 outputs. Shape-derived capacity is `N+(E+1)*63`; padded route slots contain N and trailing block owners contain -1, an invalid-expert sentinel permitted by the contract. The independent validator checks membership within each expert, all padding/tails, all block owners, count, shapes and dtypes; it does not require stable ordering inside an expert.

The call to `full(ids,e,h=0)` lies entirely inside each profiler window. Each baseline window has 27 named device tasks, including AICPU SearchSorted; each candidate window has one AI Core task. Reported time is the union of AI Core and AICPU execution intervals, counting overlap once. Host delays, device gaps and the 1 GiB cache flush are excluded; the flush and synchronization precede each window. The workload runs three warmups and 15 captures per implementation per case. No sample was dropped.

## Environment and qualification limits

- Candidate: `b691bc4f103d81fb5ab47d7ca42618232736b5dd`; measured TileOPs base `e7b1e901d22379ebd226299c597f01c204c5622d`.
- TileLang main `d50de2d3116ea84be189c581348c29d7c2483c3e`, CANN 9.3.0, torch 2.13.0+cpu, torch_npu 2.13.0, Python 3.12.3.
- Ascend950DT_9592: physical 36 Cube / 72 Vector, explicitly limited to 32 / 64 with readback for both implementations. All before/after component frequency observations were 1650/1650 MHz; this does not prove continuous frequency locking.
- Twelve cases pass the complete validator on both paths. Four terminal guard receipts (accuracy plus three performance stages) report success, no conflict and no visibility loss.

The 1.6524–5.3787x ratios compare paired measurements in this environment. They must not be recomputed by dividing these CANN 9.3 baseline times by the maintainer's CANN 9.2 candidate times. Formal acceptance still needs a complete baseline qualified for the formal environment, matching input contract and timing scope. This bundle also does not prove that this composition is the fastest available same-contract implementation. Existing matching formal-environment baseline evidence should be checked first; this publication does not request or launch another device run.
