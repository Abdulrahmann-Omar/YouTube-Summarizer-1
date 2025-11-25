# Contributing to YouTube Summarizer

Thank you for considering contributing to YouTube Summarizer! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** following the README
4. **Create a new branch** for your feature/fix

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

**Python (Backend):**
- Follow PEP 8 style guide
- Use type hints for function parameters and return values
- Add docstrings to all public functions and classes
- Maximum line length: 100 characters
- Use meaningful variable and function names

**JavaScript/React (Frontend):**
- Follow ESLint configuration provided
- Use functional components with hooks
- Keep components small and focused
- Use meaningful component and variable names
- Add comments for complex logic

### Project Structure

**Backend:**
- Place new services in `backend/services/`
- Add data models in `backend/models/schemas.py`
- Utility functions go in `backend/utils/`
- Keep main.py focused on routing and middleware

**Frontend:**
- Place new components in `frontend/src/components/`
- Add CSS in corresponding component CSS file
- Use the provided API service for backend calls
- Keep state management simple with React hooks

### Adding New Features

1. **Summarization Methods:**
   - Add method to `backend/services/summarization_service.py`
   - Update `SummarizationMethod` enum in `backend/models/schemas.py`
   - Add option to frontend in `SummarizationOptions.jsx`

2. **API Endpoints:**
   - Add endpoint in `backend/main.py`
   - Create Pydantic models in `backend/models/schemas.py`
   - Update frontend API service in `frontend/src/services/api.js`

3. **UI Components:**
   - Create component in `frontend/src/components/`
   - Add corresponding CSS file
   - Import and use in parent component

### Testing

Before submitting a PR:

1. **Test Backend:**
```bash
cd backend
source venv/bin/activate
python -m pytest  # If tests are added
```

2. **Test Frontend:**
```bash
cd frontend
npm run lint
npm run build  # Ensure it builds without errors
```

3. **Manual Testing:**
   - Test with various YouTube videos
   - Try all summarization methods
   - Test Q&A chat functionality
   - Verify responsive design on mobile
   - Test both light and dark modes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add hybrid summarization method
fix: Resolve WebSocket connection timeout
docs: Update installation instructions
style: Format code with black
refactor: Simplify NLP preprocessing pipeline
test: Add unit tests for summarization service
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Pull Request Process

1. **Update documentation** if needed
2. **Add/update tests** if applicable
3. **Ensure all tests pass**
4. **Update README** if adding major features
5. **Create pull request** with clear description

**PR Description Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How was this tested?

## Screenshots
If applicable, add screenshots

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

## Feature Requests

Have an idea for a new feature?

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with the `enhancement` label
3. **Describe the feature** clearly:
   - What problem does it solve?
   - How should it work?
   - Any implementation ideas?

## Bug Reports

Found a bug?

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with the `bug` label
3. **Include details**:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - Environment (OS, Python version, Node version)
   - Error messages/logs

## Code Review

All PRs will be reviewed for:
- Code quality and style
- Functionality and correctness
- Performance implications
- Security considerations
- Documentation completeness
- Test coverage

## Questions?

- Open an issue with the `question` label
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to YouTube Summarizer! 🎉

