from pathlib import Path
import anthropic
from typing import List, Dict, Set
import os
from dotenv import load_dotenv

class NotionTagger:
    def __init__(self) -> None:
        """Initialize the NotionTagger with Claude client and base tags."""
        load_dotenv()
        self.client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.base_tags = {
            '[[Artificial Intelligence]]', '[[Computer Science]]', '[[writing]]',
            '[[book]]', '[[boot.dev]]', '[[general]]', '[[Internet Subculture]]',
            '[[Journal]]', '[[Parasocial Grief]]', '[[Personal Writing]]',
            '[[Personal]]', '[[Scraping]]', '[[türkçe]]', '[[work]]', '[[yzf]]',
            '[[Aiden]]'
        }

    def analyze_with_claude(self, content: str) -> Set[str]:
        """
        Use Claude to analyze content and suggest tags.
        
        Args:
            content (str): The content to analyze
            
        Returns:
            Set[str]: Set of relevant tags including new suggestions
        """
        prompt = f"""
        Analyze this content and:
        1. Select relevant tags from this list: {self.base_tags}
        2. Suggest new tags that would be relevant (use the same [[tag]] format)
        3. Determine if this is a reading (article/book) or personal writing
        
        Content to analyze:
        {content[:1000]}...
        
        Respond in this format:
        EXISTING_TAGS: tag1, tag2, tag3
        NEW_TAGS: tag1, tag2, tag3
        CONTENT_TYPE: READING or WRITING
        """

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0,
            system="You are a content analyzer that suggests relevant tags for markdown files.",
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse Claude's response
        response_text = response.content[0].text
        tags = set()
        
        for line in response_text.split('\n'):
            if line.startswith(('EXISTING_TAGS:', 'NEW_TAGS:')):
                tags.update(tag.strip() for tag in line.split(':', 1)[1].split(',') if tag.strip())
            if line.startswith('CONTENT_TYPE:'):
                content_type = line.split(':', 1)[1].strip()
                tags.add(f"[[{content_type}]]")
        
        return tags

    def update_file_with_tags(self, file_path: Path) -> None:
        """
        Updates the given file by adding relevant tags at the top.
        
        Args:
            file_path (Path): Path to the file to update
        """
        content = file_path.read_text(encoding='utf-8')
        
        if content.startswith('Tags:'):
            return
        
        tags = self.analyze_with_claude(content)
        tags_line = f"Tags: {' '.join(sorted(tags))}\n\n"
        
        file_path.write_text(tags_line + content, encoding='utf-8')
        print(f"Updated {file_path} with tags: {' '.join(sorted(tags))}")

def main() -> None:
    """
    Main function to process all markdown files in the current directory.
    """
    tagger = NotionTagger()
    current_dir = Path('.')
    md_files = current_dir.glob('*.md')
    
    for file_path in md_files:
        try:
            tagger.update_file_with_tags(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    main() 