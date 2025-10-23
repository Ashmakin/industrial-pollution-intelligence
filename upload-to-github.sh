




echo "🚀 Industrial Pollution Intelligence System - GitHub Upload"
echo "=========================================================="


if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository. Please run 'git init' first."
    exit 1
fi


if ! git remote get-url origin >/dev/null 2>&1; then
    echo "📝 Please set the GitHub repository URL first:"
    echo ""
    echo "git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git"
    echo ""
    echo "Replace YOUR_USERNAME and YOUR_REPOSITORY_NAME with your actual GitHub username and repository name."
    echo ""
    read -p "Press Enter after setting the remote origin..."
fi


if ! git remote get-url origin >/dev/null 2>&1; then
    echo "❌ Error: No remote origin set. Please add the GitHub repository URL first."
    exit 1
fi

echo "✅ Git repository ready"
echo "📦 Pushing code to GitHub..."


git add .


git commit -m "feat: Complete Industrial Pollution Intelligence System

- Rust + Axum backend with PostgreSQL integration
- React + TypeScript frontend with modern UI
- Python ML pipeline with PyTorch/TensorFlow
- Advanced analytics: LSTM, CNN-LSTM, Transformer models
- Real-time CNEMC data collection with deduplication
- Interactive China map visualization with D3.js
- Comprehensive time series analysis and forecasting
- Network analysis and causal inference
- Automated data collection scheduler
- Professional dashboard and reporting system

Features:
✅ High-performance Rust backend
✅ Modern React frontend with Tailwind CSS
✅ Advanced ML models using PyTorch
✅ Real-time data collection from CNEMC API
✅ Interactive D3.js visualizations
✅ Automated data collection scheduler
✅ Comprehensive analysis tools
✅ Professional reporting system"


echo "🚀 Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Successfully uploaded to GitHub!"
echo "🌐 Your repository is now available at:"
git remote get-url origin
echo ""
echo "📋 Next steps:"
echo "1. Visit your GitHub repository"
echo "2. Add a description and topics"
echo "3. Enable GitHub Pages if you want to host the frontend"
echo "4. Set up GitHub Actions for CI/CD (optional)"
echo "5. Add collaborators if needed"
echo ""
echo "🎉 Your Industrial Pollution Intelligence System is now on GitHub!"
