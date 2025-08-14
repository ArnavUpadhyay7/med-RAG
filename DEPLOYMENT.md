# 🚀 Deployment Checklist

## Prerequisites

- [ ] GitHub repository created and code pushed
- [ ] Pinecone account and API key
- [ ] Groq account and API key
- [ ] Python 3.8+ environment

## Environment Setup

Create a `.env` file in your project root:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=medical-rag
```

## Local Testing

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the application**
   ```bash
   streamlit run app.py
   ```

3. **Verify functionality**
   - Upload a test PDF
   - Run a test query
   - Check evaluation suite

## Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit: Medical RAG system"
   git push origin main
   ```

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set entry point: `app.py`

3. **Configure Secrets**
   In Streamlit Cloud dashboard, add these secrets:
   ```
   GROQ_API_KEY = your_groq_api_key_here
   PINECONE_API_KEY = your_pinecone_api_key_here
   PINECONE_INDEX = medical-rag
   ```

4. **Deploy**
   - Click "Deploy!"
   - Wait for build to complete
   - Test the deployed application

## Hugging Face Spaces Deployment

1. **Create Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" SDK
   - Name your space

2. **Upload Code**
   - Clone the space repository
   - Copy your project files
   - Push to the space repository

3. **Set Environment Variables**
   - Go to Settings → Repository Secrets
   - Add the same environment variables as above

4. **Deploy**
   - Push changes to trigger automatic deployment
   - Monitor build logs for any issues

## Post-Deployment Verification

- [ ] Application loads without errors
- [ ] PDF upload functionality works
- [ ] Query and retrieval functions properly
- [ ] Evaluation suite runs successfully
- [ ] API keys are properly secured
- [ ] Performance is acceptable

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are in `requirements.txt`
2. **API key errors**: Verify environment variables are set correctly
3. **Pinecone connection**: Check index name and API key validity
4. **Memory issues**: Consider reducing chunk size for large documents

### Performance Optimization

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your documents
- Monitor Pinecone usage and costs
- Consider caching for frequently accessed embeddings

## Security Notes

- Never commit API keys to version control
- Use environment variables for all sensitive data
- Regularly rotate API keys
- Monitor API usage and costs

## Support Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Groq Documentation](https://console.groq.com/docs)
- [Sentence Transformers](https://www.sbert.net/)

---

**Remember**: Test thoroughly in a local environment before deploying to production!
