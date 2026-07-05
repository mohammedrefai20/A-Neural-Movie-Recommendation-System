## 9. Phase 2 Summary

✅ Loaded Phase 1 processed data (train/val/test + mappings)
✅ Built the **NCF (Neural Matrix Factorization)** architecture:
   - Embedding layers for users & movies (MF branch + MLP branch)
   - MLP with ReLU + Dropout for non-linear interaction modeling
   - Combined output layer
✅ Trained with **Adam optimizer + MSE loss + Early Stopping**
✅ Evaluated on test set → **RMSE & MAE** reported
✅ Visualized training curves and predicted-vs-actual ratings
✅ Saved the model (`ncf_model.pt`) and config (`ncf_config.pkl`) for inference
✅ Built and tested:
   - `predict_for_user()` → single predictions
   - `get_top_n_recommendations()` → **this is exactly what the User Page dashboard needs!**

### 📦 Output Files (in `models/` folder)
| File | Description |
|------|-------------|
| `ncf_model.pt` | Final trained model weights |
| `ncf_best_checkpoint.pt` | Best checkpoint during training (lowest val loss) |
| `ncf_config.pkl` | Architecture config + test metrics, needed to reload the model |

---
**Next Phase →** Phase 3: Item Similarity Module (for the Item Page — "movies similar to X").
