# Release checklist

- [x] Compact semantic/artifact/fusion checkpoints included.
- [x] Checkpoint metadata and required load order documented.
- [x] Exact 284,568-image manifest and SHA-256 hashes included.
- [x] Local absolute dataset root removed from the public audit JSON.
- [x] Final 224×224 artifact configuration recorded; 320×320 development command is not used.
- [x] Final semantic checkpoint recorded as epoch 5 with patch shuffle disabled.
- [x] Final fusion checkpoint recorded as epoch 4.
- [x] `results/` reduced to three paper-level CSV summaries; generated outputs are excluded from Git.
- [x] Ablation launch/config records moved to `experiments/`.
- [x] Six supplied result archives were audited; stale snapshot metrics were excluded.
- [ ] Choose and add a LICENSE before a normal public open-source release.
- [ ] Add final paper citation after publication/de-anonymization.
- [ ] Optionally move `.pth` files to GitHub Releases/Hugging Face if you want a lighter Git history.
- [ ] Re-run `bash scripts/05_eval_rider.sh` on a clean environment before tagging v1.0.
