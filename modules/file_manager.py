"""
File Manager for Telegram message analysis.

This service handles:
- Directory creation and management
- File writing and I/O operations
- File cleanup and maintenance
- Error handling for file operations
"""

import logging
import os
import glob
from typing import List, Optional, Dict
from datetime import datetime


class FileManager:
    """Service for managing file operations and directory structure."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def ensure_directory_exists(self, directory: str) -> bool:
        """Ensure output directory exists, create if necessary."""
        try:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory}: {e}")
            return False
    
    def write_html_file(self, filename: str, content: str) -> Optional[str]:
        """Write HTML content to file."""
        try:
            file_path = os.path.join(self.output_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Successfully wrote HTML file: {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to write HTML file {filename}: {e}")
            return None
    
    def write_multiple_files(self, files: List[tuple]) -> Dict[str, bool]:
        """Write multiple HTML files at once.
        
        Args:
            files: List of (filename, content) tuples
            
        Returns:
            Dict mapping filenames to success status
        """
        results = {}
        
        for filename, content in files:
            success = self.write_html_file(filename, content) is not None
            results[filename] = success
            
            if not success:
                self.logger.error(f"Failed to write file: {filename}")
        
        return results
    
    def cleanup_old_files(self, pattern: str = "*.html", max_age_hours: int = 24) -> int:
        """Remove old generated files based on pattern and age."""
        try:
            search_pattern = os.path.join(self.output_dir, pattern)
            files_to_clean = glob.glob(search_pattern)
            
            cleaned_count = 0
            current_time = datetime.now()
            
            for file_path in files_to_clean:
                try:
                    file_age = current_time - datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_age.total_seconds() > (max_age_hours * 3600):
                        os.remove(file_path)
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old file: {file_path}")
                        
                except Exception as e:
                    self.logger.warning(f"Could not process file {file_path}: {e}")
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old files")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during file cleanup: {e}")
            return 0
    
    def get_file_info(self, filename: str) -> Optional[Dict]:
        """Get information about a specific file."""
        try:
            file_path = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            
            return {
                'filename': filename,
                'size_bytes': stat.st_size,
                'size_kb': round(stat.st_size / 1024, 2),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'path': file_path
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {filename}: {e}")
            return None
    
    def list_generated_files(self, pattern: str = "*.html") -> List[Dict]:
        """List all generated files with their information."""
        try:
            search_pattern = os.path.join(self.output_dir, pattern)
            files = glob.glob(search_pattern)
            
            file_info_list = []
            for file_path in files:
                filename = os.path.basename(file_path)
                info = self.get_file_info(filename)
                if info:
                    file_info_list.append(info)
            
            return file_info_list
            
        except Exception as e:
            self.logger.error(f"Error listing generated files: {e}")
            return []
    
    def validate_output_directory(self) -> bool:
        """Validate that the output directory is accessible and writable."""
        try:
            # Check if directory exists
            if not os.path.exists(self.output_dir):
                return self.ensure_directory_exists(self.output_dir)
            
            # Check if directory is writable
            test_file = os.path.join(self.output_dir, '.test_write')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                return True
            except Exception:
                self.logger.error(f"Output directory {self.output_dir} is not writable")
                return False
                
        except Exception as e:
            self.logger.error(f"Error validating output directory: {e}")
            return False
    
    def get_directory_size(self) -> int:
        """Get total size of output directory in bytes."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.output_dir):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            return total_size
        except Exception as e:
            self.logger.error(f"Error calculating directory size: {e}")
            return 0
