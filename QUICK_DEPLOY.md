# ⚡ Quick Deploy Guide - Streamlit Cloud

## 🚀 Deploy in 3 Steps (5 minutes!)

### Step 1: Verify Your Repository
✅ Make sure these files are in your GitHub repo:
- `src/app.py`
- `src/gradcam.py`
- `src/dataset.py`
- `src/utils.py`
- `models/best_model.pth`
- `requirements.txt`

### Step 2: Deploy on Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Click**: "Sign in" (use GitHub account)
3. **Click**: "New app"
4. **Fill in**:
   - **Repository**: `AlphaDoc1/canser_detection-`
   - **Branch**: `main`
   - **Main file path**: `src/app.py`
5. **Click**: "Deploy"

### Step 3: Wait & Share! 🎉

Your app will be live at:
```
https://your-app-name.streamlit.app
```

**That's it!** No configuration needed.

---

## 🔧 If You Get Errors

### Error: Model not found
**Fix**: Make sure `models/best_model.pth` is committed to GitHub
```bash
git add models/best_model.pth
git commit -m "Add model file"
git push
```

### Error: Import errors
**Fix**: Check `requirements.txt` has all dependencies

### Error: Out of memory
**Fix**: This is normal on free tier - first load takes time

---

## 📱 Your Live App Features

Once deployed, users can:
- ✅ Upload blood cell images
- ✅ Get instant predictions (ALL vs HEM)
- ✅ See confidence scores
- ✅ View Grad-CAM heatmaps
- ✅ See probability breakdowns

---

## 🎯 Alternative: Hugging Face Spaces

If Streamlit Cloud doesn't work:

1. Go to: https://huggingface.co/spaces
2. Create new Space
3. Select "Streamlit" SDK
4. Upload your files
5. Deploy!

---

**Recommended**: Start with Streamlit Cloud - it's the easiest! 🚀


