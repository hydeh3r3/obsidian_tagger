# Obsidian File Tagger

A Python utility to automatically tag and organize files in Obsidian using AI-powered content analysis.

## Overview

This tool analyzes files in your Notion workspace and automatically suggests relevant tags based on the content. It uses Claude AI to understand document content and generate appropriate tags.

> **Note:** The autoencoder component is currently under development and not yet functional. The current version uses direct API calls to Claude for content analysis.

## Features

- Analyzes text content of Notion pages and documents
- Generates relevant tags based on content analysis
- Updates Notion pages with suggested tags
- Configurable tagging rules and categories

## Requirements

- Python 3.8+
- Notion API access
- Claude API access
- Required Python packages (see requirements.txt)

## Setup

1. Install dependencies:
Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure your API keys:
   - Copy `.env.example` to `.env`
   - Add your Anthropic API key to `.env`
```bash
cp .env.example .env
# Edit .env with your API key
```

3. Run the script:

```bash
python tag_notion_files_claude.py
```

or 

```bash
python tag_notion_files_autoencoder.py 
```

## License

MIT License - see LICENSE file for details

## Areas Needing Help

- Autoencoder implementation for more efficient content analysis
- Additional tag categories and rules
- Performance optimizations
- Test coverage
- Documentation improvements

## Pull Request Process

1. Create a feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit PR with clear description

## Code Style

- Use type hints
- Include docstrings
- Follow PEP 8
- Add unit tests

## Questions?

Open an issue for any questions about contributing.
