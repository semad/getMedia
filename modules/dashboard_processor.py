import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from jinja2 import Environment, FileSystemLoader

from config import (
    DASHBOARD_INPUT_DIR, DASHBOARD_OUTPUT_DIR,
    DASHBOARD_INDEX_FILENAME, DASHBOARD_CSS_FILENAME,
    DASHBOARD_JS_FILENAME, DASHBOARD_DATA_FILENAME,
    DASHBOARD_CSS_PATH, DASHBOARD_JS_PATH,
    DASHBOARD_GA_MEASUREMENT_ID, DASHBOARD_GA_ENABLED,
    DASHBOARD_SUPPORTED_ANALYSIS_TYPES, DASHBOARD_SUPPORTED_SOURCE_TYPES,
    DASHBOARD_MAX_CHANNEL_NAME_LENGTH, DASHBOARD_CHARTJS_CDN_URL,
    TEMPLATES_DIR, ANALYSIS_BASE, DASHBOARDS_DIR
)


class DashboardProcessor:
    """Complete dashboard processor handling all dashboard functionality."""

    def __init__(self, input_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 channels: Optional[str] = None,
                 verbose: bool = False):
        """Initialize the dashboard processor."""
        self.input_dir = input_dir or DASHBOARD_INPUT_DIR
        self.output_dir = Path(output_dir or DASHBOARD_OUTPUT_DIR)
        self.channels = self._parse_channels(channels)
        self.verbose = verbose

        self.logger = self._setup_logger()
        self.template_env = self._setup_templates()

    def _setup_logger(self) -> logging.Logger:
        """Configure logging for the dashboard processor."""
        logger = logging.getLogger('dashboard_processor')
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _setup_templates(self) -> Environment:
        """Setup Jinja2 template environment."""
        template_dir = Path(TEMPLATES_DIR) / "dashboard"
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=True
        )
        
        # Add custom filters
        env.filters['tojson'] = self._tojson_filter
        return env
    
    def _tojson_filter(self, obj):
        """Custom Jinja2 filter to convert Python objects to JSON."""
        import json
        return json.dumps(obj, default=str)
    
    def _transform_data_for_charts(self, data: Dict[str, Any], report_type: str) -> Dict[str, Any]:
        """Transform analysis data into Chart.js compatible format."""
        try:
            if report_type == 'filename_analysis':
                return self._transform_filename_analysis(data)
            elif report_type == 'filesize_analysis':
                return self._transform_filesize_analysis(data)
            elif report_type == 'message_analysis':
                return self._transform_message_analysis(data)
            else:
                return data
        except Exception as e:
            self.logger.error(f"Error transforming {report_type} data: {e}")
            return data
    
    def _transform_filename_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform filename analysis data for charts."""
        # Extract data from nested structure
        duplicate_detection = data.get('duplicate_filename_detection', {})
        pattern_analysis = data.get('filename_pattern_analysis', {})
        
        unique_files = duplicate_detection.get('total_unique_filenames', 0)
        duplicate_files = duplicate_detection.get('files_with_duplicate_names', 0)
        
        # Create a pie chart for file uniqueness
        return {
            'type': 'pie',
            'data': {
                'labels': ['Unique Files', 'Duplicate Files'],
                'datasets': [{
                    'label': 'File Uniqueness',
                    'data': [unique_files, duplicate_files],
                    'backgroundColor': ['#28a745', '#dc3545'],
                    'borderWidth': 2
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'aspectRatio': 1,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'File Uniqueness Distribution'
                    },
                    'legend': {
                        'position': 'bottom'
                    }
                }
            }
        }
    
    def _transform_filesize_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform filesize analysis data for charts."""
        # Extract data from nested structure
        filesize_distribution = data.get('filesize_distribution_analysis', {})
        size_distribution = filesize_distribution.get('size_frequency_distribution', {})
        
        labels = []
        values = []
        colors = []
        
        # Map size categories to colors
        color_map = {
            '0-1MB': '#17a2b8',
            '1-5MB': '#28a745', 
            '5-10MB': '#ffc107',
            '10MB+': '#dc3545'
        }
        
        for size_category, count in size_distribution.items():
            labels.append(size_category)
            values.append(count)
            colors.append(color_map.get(size_category, '#6c757d'))
        
        return {
            'type': 'bar',
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': 'Number of Files',
                    'data': values,
                    'backgroundColor': colors,
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'File Size Distribution'
                    },
                    'legend': {
                        'display': False
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            }
        }
    
    def _transform_file_types_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform file types analysis data for charts."""
        # Extract file extensions data from nested structure
        filename_pattern_analysis = data.get('filename_pattern_analysis', {})
        common_extensions = filename_pattern_analysis.get('common_extensions', [])
        
        if not common_extensions:
            return {
                'type': 'bar',
                'data': {
                    'labels': ['No Data'],
                    'datasets': [{
                        'label': 'No File Types Available',
                        'data': [0],
                        'backgroundColor': '#6c757d'
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'File Types - No Data Available'
                        }
                    }
                }
            }
        
        # Get top 10 file types
        top_extensions = common_extensions[:10]
        
        labels = []
        values = []
        colors = []
        
        # Color map for different file types
        color_map = {
            '.pdf': '#dc3545',    # Red for PDFs
            '.epub': '#007bff',   # Blue for EPUBs
            '.mp3': '#28a745',    # Green for MP3s
            '.mp4': '#ffc107',    # Yellow for MP4s
            '.zip': '#6c757d',    # Gray for ZIPs
            '.rar': '#6c757d',    # Gray for RARs
            '.apk': '#17a2b8',    # Cyan for APKs
            '.png': '#fd7e14',    # Orange for PNGs
            '.jpg': '#fd7e14',    # Orange for JPGs
            '.jpeg': '#fd7e14',   # Orange for JPEGs
        }
        
        for ext_data in top_extensions:
            ext = ext_data.get('ext', '')
            count = ext_data.get('count', 0)
            
            labels.append(ext)
            values.append(count)
            colors.append(color_map.get(ext, '#6c757d'))  # Default gray for unknown types
        
        return {
            'type': 'bar',
            'data': {
                'labels': labels,
                'datasets': [{
                    'label': 'Number of Files',
                    'data': values,
                    'backgroundColor': colors,
                    'borderColor': colors,
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'File Types Distribution'
                    },
                    'legend': {
                        'display': False
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    },
                    'x': {
                        'ticks': {
                            'maxRotation': 45
                        }
                    }
                }
            }
        }

    def _transform_message_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform message analysis data for charts."""
        # Extract pattern recognition data
        pattern_recognition = data.get('pattern_recognition', {})
        hashtags = pattern_recognition.get('hashtags', {})
        emojis = pattern_recognition.get('emojis', {})
        
        # Create a bar chart for top hashtags
        top_hashtags = hashtags.get('top_hashtags', [])[:10]  # Top 10 hashtags
        
        if top_hashtags:
            labels = [tag['tag'] for tag in top_hashtags]
            counts = [tag['count'] for tag in top_hashtags]
            
            return {
                'type': 'bar',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Hashtag Usage',
                        'data': counts,
                        'backgroundColor': '#007bff',
                        'borderColor': '#0056b3',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Top Hashtags'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True
                        },
                        'x': {
                            'ticks': {
                                'maxRotation': 45
                            }
                        }
                    }
                }
            }
        else:
            # Fallback to emoji chart if no hashtags
            top_emojis = emojis.get('top_emojis', [])[:10]  # Top 10 emojis
            
            if top_emojis:
                labels = [emoji['emoji'] for emoji in top_emojis]
                counts = [emoji['count'] for emoji in top_emojis]
                
                return {
                    'type': 'bar',
                    'data': {
                        'labels': labels,
                        'datasets': [{
                            'label': 'Emoji Usage',
                            'data': counts,
                            'backgroundColor': '#28a745',
                            'borderColor': '#1e7e34',
                            'borderWidth': 1
                        }]
                    },
                    'options': {
                        'responsive': True,
                        'plugins': {
                            'title': {
                                'display': True,
                                'text': 'Top Emojis'
                            },
                            'legend': {
                                'display': False
                            }
                        },
                        'scales': {
                            'y': {
                                'beginAtZero': True
                            }
                        }
                    }
                }
            else:
                # Fallback to empty chart
                return {
                    'type': 'bar',
                    'data': {
                        'labels': ['No Data'],
                        'datasets': [{
                            'label': 'No Data Available',
                            'data': [0],
                            'backgroundColor': '#6c757d'
                        }]
                    },
                    'options': {
                        'responsive': True,
                        'plugins': {
                            'title': {
                                'display': True,
                                'text': 'Message Analysis - No Data Available'
                            }
                        }
                    }
                }

    def _transform_timeline_data(self, timeline_data: Dict[str, int]) -> Dict[str, Any]:
        """Transform timeline data into a line chart format."""
        try:
            # Sort dates and prepare data
            sorted_dates = sorted(timeline_data.keys())
            labels = []
            values = []
            
            # Limit to reasonable number of data points for performance
            max_points = 100
            if len(sorted_dates) > max_points:
                # Sample the data to avoid too many points
                step = len(sorted_dates) // max_points
                sampled_dates = sorted_dates[::step]
            else:
                sampled_dates = sorted_dates
            
            for date in sampled_dates:
                labels.append(date)
                values.append(timeline_data[date])
            
            return {
                'type': 'line',
                'data': {
                    'labels': labels,
                    'datasets': [{
                        'label': 'Messages per Day',
                        'data': values,
                        'borderColor': '#dc3545',
                        'backgroundColor': 'rgba(220, 53, 69, 0.1)',
                        'borderWidth': 2,
                        'fill': True,
                        'tension': 0.1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Message Activity Timeline'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'x': {
                            'title': {
                                'display': True,
                                'text': 'Date'
                            }
                        },
                        'y': {
                            'beginAtZero': True,
                            'title': {
                                'display': True,
                                'text': 'Messages'
                            }
                        }
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error transforming timeline data: {e}")
            return {}

    def _transform_monthly_histogram(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform monthly message data for histogram chart."""
        try:
            temporal_patterns = data.get('temporal_patterns', {})
            monthly_data = temporal_patterns.get('monthly', {})
            
            # Create monthly labels and data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            counts = []
            
            for month_num in range(1, 13):
                counts.append(monthly_data.get(str(month_num), 0))
            
            return {
                'type': 'bar',
                'data': {
                    'labels': months,
                    'datasets': [{
                        'label': 'Messages per Month',
                        'data': counts,
                        'backgroundColor': '#17a2b8',
                        'borderColor': '#138496',
                        'borderWidth': 1
                    }]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Monthly Message Distribution'
                        },
                        'legend': {
                            'display': False
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True
                        }
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error transforming monthly histogram: {e}")
            return {}

    def _transform_source_breakdown(self, source_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Transform source breakdown data for charts."""
        try:
            labels = []
            message_data = []
            file_data = []
            colors = ['#28a745', '#007bff', '#ffc107']  # Green, Blue, Yellow
            
            for i, (source_type, data) in enumerate(source_breakdown.items()):
                # Format source type names for display
                display_name = source_type.replace('_', ' ').title()
                labels.append(display_name)
                message_data.append(data.get('messages', 0))
                file_data.append(data.get('files', 0))
            
            return {
                'type': 'bar',
                'data': {
                    'labels': labels,
                    'datasets': [
                        {
                            'label': 'Messages',
                            'data': message_data,
                            'backgroundColor': colors,
                            'borderColor': colors,
                            'borderWidth': 1
                        },
                        {
                            'label': 'Files',
                            'data': file_data,
                            'backgroundColor': [f"{color}80" for color in colors],  # Add transparency
                            'borderColor': colors,
                            'borderWidth': 1
                        }
                    ]
                },
                'options': {
                    'responsive': True,
                    'plugins': {
                        'title': {
                            'display': True,
                            'text': 'Data Source Breakdown'
                        },
                        'legend': {
                            'display': True,
                            'position': 'top'
                        }
                    },
                    'scales': {
                        'y': {
                            'beginAtZero': True
                        }
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error transforming source breakdown: {e}")
            return {}

    def _get_timeline_data_for_channel(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Get timeline data for a specific channel from collection files."""
        try:
            # Find the collection file for this channel
            collections_dir = Path("reports/0_collections")
            collection_files = list(collections_dir.glob(f"*{channel_name}*.json"))
            
            if not collection_files:
                self.logger.warning(f"Collection file not found for {channel_name}")
                return None
            
            # Process all collection files for this channel
            all_dates = []
            for collection_file in collection_files:
                with open(collection_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data or len(data) == 0:
                    continue
                
                # Extract messages from the collection data
                messages = data[0].get('messages', [])
                if not messages:
                    continue
                
                # Process dates
                for message in messages:
                    if 'date' in message:
                        try:
                            # Parse the date string
                            date_str = message['date']
                            if isinstance(date_str, str):
                                # Handle different date formats
                                if '+' in date_str:
                                    date_str = date_str.split('+')[0]  # Remove timezone
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                all_dates.append(date_obj)
                        except Exception as e:
                            self.logger.debug(f"Error parsing date {date_str}: {e}")
                            continue
            
            if not all_dates:
                return None
            
            # Sort dates
            all_dates.sort()
            
            # Create timeline data (daily aggregation)
            timeline_data = {}
            for date in all_dates:
                date_key = date.strftime('%Y-%m-%d')
                timeline_data[date_key] = timeline_data.get(date_key, 0) + 1
            
            # Convert to chart format
            return self._transform_timeline_data(timeline_data)
            
        except Exception as e:
            self.logger.error(f"Error getting timeline data for {channel_name}: {e}")
            return None

    def _get_monthly_histogram_for_channel(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Get monthly histogram data for a specific channel from collection files."""
        try:
            # Find the collection file for this channel
            collections_dir = Path("reports/0_collections")
            collection_files = list(collections_dir.glob(f"*{channel_name}*.json"))
            
            if not collection_files:
                self.logger.warning(f"Collection file not found for {channel_name}")
                return None
            
            # Process all collection files for this channel
            all_dates = []
            for collection_file in collection_files:
                with open(collection_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not data or len(data) == 0:
                    continue
                
                # Extract messages from the collection data
                messages = data[0].get('messages', [])
                if not messages:
                    continue
                
                # Process dates
                for message in messages:
                    if 'date' in message:
                        try:
                            # Parse the date string
                            date_str = message['date']
                            if isinstance(date_str, str):
                                # Handle different date formats
                                if '+' in date_str:
                                    date_str = date_str.split('+')[0]  # Remove timezone
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                                all_dates.append(date_obj)
                        except Exception as e:
                            self.logger.debug(f"Error parsing date {date_str}: {e}")
                            continue
            
            if not all_dates:
                return None
            
            # Create monthly data
            monthly_data = {}
            for date in all_dates:
                month_key = date.strftime('%Y-%m')
                monthly_data[month_key] = monthly_data.get(month_key, 0) + 1
            
            # Transform the monthly data
            return self._transform_monthly_histogram({'monthly': monthly_data})
            
        except Exception as e:
            self.logger.error(f"Error getting monthly histogram for {channel_name}: {e}")
            return None

    def _parse_channels(self, channels_str: Optional[str]) -> List[str]:
        """Parse comma-separated channel list."""
        if not channels_str or channels_str.lower() == 'all':
            return []
        return [c.strip() for c in channels_str.split(',')]

    def process(self):
        """Main processing method - generates complete dashboard."""
        self.logger.info("Starting dashboard processing...")

        try:
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (self.output_dir / DASHBOARD_CSS_PATH).mkdir(parents=True, exist_ok=True)
            (self.output_dir / DASHBOARD_JS_PATH).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'data').mkdir(exist_ok=True)

            # Process data
            self.logger.info("Processing analysis data...")
            files = self._discover_analysis_files()
            dashboard_data = self._aggregate_channel_data(files)

            if not dashboard_data or not dashboard_data.get('channels'):
                self.logger.warning("No channel data found. Generating empty dashboard.")
                dashboard_data = self._create_empty_dashboard_data()

            # Generate shared files
            self._generate_shared_files()

            # Generate HTML pages
            self._generate_html_pages(dashboard_data)

            self.logger.info(f"Dashboard generated successfully in {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Critical error during dashboard processing: {e}")
            raise

    # Data Processing Methods
    def _discover_analysis_files(self) -> Dict[str, List[Path]]:
        """Find all analysis JSON files in the input directory."""
        files = {}
        input_path = Path(self.input_dir)
        
        if not input_path.exists():
            self.logger.warning(f"Input directory does not exist: {input_path}")
            return files
        
        # First, look for files in subdirectories (original behavior)
        for source_type in DASHBOARD_SUPPORTED_SOURCE_TYPES:
            source_path = input_path / source_type
            if source_path.exists():
                files[source_type] = list(source_path.rglob("*.json"))
                self.logger.debug(f"Found {len(files[source_type])} files in {source_type}")
        
        # If no files found in subdirectories, look for channel-specific directories
        if not any(files.values()):
            self.logger.info("No files found in subdirectories, looking for channel-specific directories")
            # Look for channel directories (e.g., books/, books_magazine/, etc.)
            channel_dirs = [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            if channel_dirs:
                # Collect all JSON files from channel directories
                all_channel_files = []
                for channel_dir in channel_dirs:
                    channel_files = list(channel_dir.glob("*.json"))
                    all_channel_files.extend(channel_files)
                    self.logger.debug(f"Found {len(channel_files)} files in channel directory: {channel_dir.name}")
                
                if all_channel_files:
                    files["file_messages"] = all_channel_files
                    self.logger.debug(f"Found {len(all_channel_files)} files in channel directories")
        
        # If still no files found, look directly in input directory
        if not any(files.values()):
            self.logger.info("No files found in channel directories, looking directly in input directory")
            direct_files = list(input_path.glob("*.json"))
            if direct_files:
                files["file_messages"] = direct_files
                self.logger.debug(f"Found {len(direct_files)} files directly in input directory")
        
        return files

    def _load_analysis_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate JSON data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle our analysis file format (array with single object)
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            
            # Validate required fields for our format
            if not isinstance(data, dict):
                self.logger.warning(f"Invalid data format in {file_path}")
                return None
            
            # Check for our analysis file format
            if 'channel_name' in data and 'analysis_type' in data:
                return data
            
            # Check for comprehensive analysis format (array with single object)
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                analysis_obj = data[0]
                if 'channel_name' in analysis_obj and 'analysis_type' in analysis_obj:
                    return analysis_obj
            
            # Check for old format
            if not all(key in data for key in ['report_type', 'generated_at']):
                # Check if this is a raw analysis data file (individual analysis files)
                if any(field in data for field in ['duplicate_filename_detection', 'filesize_distribution_analysis', 'content_statistics']):
                    # This is a raw analysis data file, add minimal metadata
                    data['analysis_type'] = self._infer_analysis_type_from_data(data)
                    data['channel_name'] = self._extract_channel_name_from_path(file_path)
                    return data
                else:
                    self.logger.warning(f"Missing required fields in {file_path}")
                    return None
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def _infer_analysis_type_from_data(self, data: Dict[str, Any]) -> str:
        """Infer analysis type from data structure."""
        if 'duplicate_filename_detection' in data or 'common_extensions' in data:
            return 'filename_analysis'
        elif 'filesize_distribution_analysis' in data or 'size_frequency_distribution' in data:
            return 'filesize_analysis'
        elif 'content_statistics' in data or 'pattern_recognition' in data:
            return 'message_analysis'
        else:
            return 'unknown'
    
    def _extract_channel_name_from_path(self, file_path: Path) -> str:
        """Extract channel name from file path."""
        # For channel-specific directories, extract from parent directory name
        if file_path.parent.name != self.input_dir.split('/')[-1]:  # Not in root analysis directory
            return file_path.parent.name
        
        # For files in root directory, extract from filename
        filename = file_path.stem
        # Remove common suffixes to get the base channel name
        suffixes_to_remove = ['_analysis', '_filename_analysis', '_filesize_analysis', '_message_analysis']
        for suffix in suffixes_to_remove:
            if filename.endswith(suffix):
                filename = filename[:-len(suffix)]
                break
        return filename

    def _extract_channel_name(self, file_path: Path) -> Optional[str]:
        """Extract channel name from file path."""
        try:
            # Path structure: .../source_type/channel_name/analysis_file.json
            parts = file_path.parts
            # Find the source type (file_messages, db_messages, diff_messages)
            for i, part in enumerate(parts):
                if part in DASHBOARD_SUPPORTED_SOURCE_TYPES:
                    if i + 1 < len(parts):
                        return parts[i + 1]
            
            # If no source type found, extract from filename directly
            # Files like "SherwinVakiliLibrary_analysis.json" -> "SherwinVakiliLibrary"
            filename = file_path.stem
            if '_' in filename:
                # Remove common suffixes like _analysis, _filename_analysis, etc.
                suffixes_to_remove = ['_analysis', '_filename_analysis', '_filesize_analysis', '_message_analysis']
                for suffix in suffixes_to_remove:
                    if filename.endswith(suffix):
                        filename = filename[:-len(suffix)]
                        break
                return filename
        except Exception as e:
            self.logger.error(f"Error extracting channel name from {file_path}: {e}")
        return None

    def _sanitize_channel_name(self, channel_name: str) -> str:
        """Clean channel name for file system use."""
        # Remove or replace unsafe characters
        sanitized = channel_name.replace('@', 'at_').replace(' ', '_')
        sanitized = ''.join(c for c in sanitized if c.isalnum() or c in '_-')
        
        # Limit length
        if len(sanitized) > DASHBOARD_MAX_CHANNEL_NAME_LENGTH:
            sanitized = sanitized[:DASHBOARD_MAX_CHANNEL_NAME_LENGTH]
        
        return sanitized

    def _validate_and_limit_data(self, data: Dict[str, Any], report_type: str) -> bool:
        """Validate and limit data size for performance."""
        try:
            # Check if data is too large
            data_str = json.dumps(data, default=str)
            if len(data_str) > 1000000:  # 1MB limit for performance
                self.logger.warning(f"Data too large for {report_type}, truncating")
                return False
            
            # Validate data structure based on report type
            if report_type == 'analysis_summary':
                # Check for either the old or new field names
                has_old_fields = all(field in data for field in ['total_messages', 'total_files'])
                has_new_fields = all(field in data for field in ['total_messages_analyzed', 'total_files_analyzed'])
                if not (has_old_fields or has_new_fields):
                    self.logger.warning(f"Missing required fields in {report_type}")
                    return False
            elif report_type == 'message_analysis':
                # Validate message analysis has basic fields
                if not any(field in data for field in ['total_messages', 'text_messages', 'media_messages']):
                    self.logger.warning(f"Missing required fields in {report_type}")
                    return False
            elif report_type == 'filename_analysis':
                # Validate filename analysis has basic fields (check nested structure)
                has_nested_fields = any(field in data for field in ['duplicate_filename_detection', 'common_extensions'])
                has_top_level_fields = any(field in data for field in ['total_files', 'unique_filenames', 'duplicate_filenames'])
                if not (has_nested_fields or has_top_level_fields):
                    self.logger.warning(f"Missing required fields in {report_type}")
                    return False
            elif report_type == 'filesize_analysis':
                # Validate filesize analysis has basic fields (check nested structure)
                has_nested_fields = any(field in data for field in ['filesize_distribution_analysis', 'size_frequency_distribution'])
                has_top_level_fields = any(field in data for field in ['total_size_bytes', 'total_size_mb', 'avg_file_size_mb'])
                if not (has_nested_fields or has_top_level_fields):
                    self.logger.warning(f"Missing required fields in {report_type}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data for {report_type}: {e}")
            return False

    def _aggregate_channel_data(self, files: Dict[str, List[Path]]) -> Dict[str, Any]:
        """Combine data from multiple files into dashboard structure."""
        dashboard_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_channels': 0,
                'total_files': 0
            },
            'summary': {
                'total_messages': 0,
                'total_files': 0,
                'channels_count': 0
            },
            'channels': {}
        }
        
        channel_data = {}
        
        for source_type, file_list in files.items():
            for file_path in file_list:
                # For individual analysis files, extract the base channel name
                if source_type == 'file_messages' and any(suffix in file_path.name for suffix in ['_filename_analysis', '_filesize_analysis', '_message_analysis']):
                    channel_name = self._extract_channel_name_from_path(file_path)
                else:
                    channel_name = self._extract_channel_name(file_path)
                
                if not channel_name:
                    continue
                
                # Filter by channels if specified
                if self.channels and channel_name not in self.channels:
                    continue
                
                data = self._load_analysis_data(file_path)
                if not data:
                    continue
                
                if channel_name not in channel_data:
                    channel_data[channel_name] = {
                        'messages': 0,
                        'files': 0,
                        'filename_analysis': {},
                        'filesize_analysis': {},
                        'message_analysis': {},
                        'analysis_summary': {},
                        'source_breakdown': {
                            'file_messages': {'messages': 0, 'files': 0},
                            'db_messages': {'messages': 0, 'files': 0},
                            'diff_messages': {'messages': 0, 'files': 0}
                        }
                    }
                
                # Process the analysis data
                if isinstance(data, dict):
                    # Handle our analysis file format
                    if 'analysis_type' in data:
                        # This is our comprehensive analysis file
                        analysis_results = data.get('analysis_results', {})
                        
                        # Process filename analysis
                        if 'filename_analysis' in analysis_results:
                            filename_data = analysis_results['filename_analysis']
                            channel_data[channel_name]['filename_analysis'] = self._transform_filename_analysis(filename_data)
                            # Also process file types from filename analysis
                            channel_data[channel_name]['file_types_analysis'] = self._transform_file_types_analysis(filename_data)
                            # Add detailed filename metrics
                            channel_data[channel_name]['filename_metrics'] = self._extract_filename_metrics(filename_data)
                        
                        # Process filesize analysis
                        if 'filesize_analysis' in analysis_results:
                            filesize_data = analysis_results['filesize_analysis']
                            channel_data[channel_name]['filesize_analysis'] = self._transform_filesize_analysis(filesize_data)
                            # Add detailed filesize metrics
                            channel_data[channel_name]['filesize_metrics'] = self._extract_filesize_metrics(filesize_data)
                        
                        # Process message analysis
                        if 'message_analysis' in analysis_results:
                            message_data = analysis_results['message_analysis']
                            channel_data[channel_name]['message_analysis'] = self._transform_message_analysis(message_data)
                            # Add detailed message metrics
                            channel_data[channel_name]['message_metrics'] = self._extract_message_metrics(message_data)
                            # Add additional message charts
                            channel_data[channel_name]['mentions_analysis'] = self._transform_mentions_analysis(message_data)
                            channel_data[channel_name]['urls_analysis'] = self._transform_urls_analysis(message_data)
                            channel_data[channel_name]['language_analysis'] = self._transform_language_analysis(message_data)
                            # Add creator analysis
                            channel_data[channel_name]['creator_analysis'] = self._transform_creator_analysis(message_data)
                            channel_data[channel_name]['creator_metrics'] = self._extract_creator_metrics(message_data)
                        
                        # Update summary data only if we have actual records
                        data_summary = data.get('data_summary', {})
                        total_records = data_summary.get('total_records', 0)
                        self.logger.debug(f"Processing comprehensive analysis for {channel_name}: {total_records} records")
                        if total_records > 0:
                            channel_data[channel_name]['messages'] = total_records
                            
                            # Get actual file count from filename analysis
                            analysis_results = data.get('analysis_results', {})
                            filename_analysis = analysis_results.get('filename_analysis', {})
                            duplicate_detection = filename_analysis.get('duplicate_filename_detection', {})
                            total_files = duplicate_detection.get('total_files', 0)
                            channel_data[channel_name]['files'] = total_files
                        
                    else:
                        # Handle individual analysis files (filename, filesize, message)
                        analysis_type = data.get('analysis_type', '')
                        if analysis_type == 'filename_analysis':
                            channel_data[channel_name]['filename_analysis'] = self._transform_filename_analysis(data)
                            # Also process file types from filename analysis
                            channel_data[channel_name]['file_types_analysis'] = self._transform_file_types_analysis(data)
                        elif analysis_type == 'filesize_analysis':
                            channel_data[channel_name]['filesize_analysis'] = self._transform_filesize_analysis(data)
                        elif analysis_type == 'message_analysis':
                            channel_data[channel_name]['message_analysis'] = self._transform_message_analysis(data)
                
                elif isinstance(data, list):
                    # Handle old format with list of reports
                    for report in data:
                        report_type = report.get('report_type', '')
                        if report_type in DASHBOARD_SUPPORTED_ANALYSIS_TYPES:
                            # Extract data from the appropriate field based on report type
                            if report_type == 'analysis_summary':
                                report_data = report.get('summary', {})
                            else:
                                # For other analysis types, the data is directly in the report object
                                # Remove metadata fields to get the actual analysis data
                                report_data = {k: v for k, v in report.items() 
                                             if k not in ['report_type', 'generated_at', 'analysis_version']}
                        
                        # Validate and limit data size for performance
                        if self._validate_and_limit_data(report_data, report_type):
                            # Transform data for chart visualization
                            chart_data = self._transform_data_for_charts(report_data, report_type)
                            channel_data[channel_name][report_type] = chart_data
                            
                            # Update summary metrics
                            if report_type == 'analysis_summary':
                                # Store source breakdown data
                                channel_data[channel_name]['source_breakdown'][source_type]['messages'] = report_data.get('total_messages_analyzed', 0)
                                channel_data[channel_name]['source_breakdown'][source_type]['files'] = report_data.get('total_files_analyzed', 0)
                                
                                # Use diff_messages as the primary data source for display
                                if source_type == 'diff_messages':
                                    channel_data[channel_name]['messages'] = report_data.get('total_messages_analyzed', 0)
                                    channel_data[channel_name]['files'] = report_data.get('total_files_analyzed', 0)
                        else:
                            self.logger.warning(f"Skipping {report_type} for {channel_name} due to validation failure")
        
        dashboard_data['channels'] = channel_data
        dashboard_data['metadata']['total_channels'] = len(channel_data)
        dashboard_data['summary']['channels_count'] = len(channel_data)
        
        # Add source breakdown charts and monthly histogram for each channel
        for channel_name, data in channel_data.items():
            if 'source_breakdown' in data:
                source_breakdown_chart = self._transform_source_breakdown(data['source_breakdown'])
                if source_breakdown_chart:
                    data['source_breakdown_chart'] = source_breakdown_chart
            
            # Add monthly histogram from message analysis data
            if 'message_analysis' in data and data['message_analysis']:
                # Extract the raw message analysis data to get monthly data
                # We need to get this from the original analysis files
                monthly_histogram = self._get_monthly_histogram_for_channel(channel_name)
                if monthly_histogram:
                    data['monthly_histogram'] = monthly_histogram
            
            # Add timeline chart from collection data
            timeline_chart = self._get_timeline_data_for_channel(channel_name)
            if timeline_chart:
                data['timeline_chart'] = timeline_chart
        
        # Calculate total summary
        for channel_data_item in channel_data.values():
            dashboard_data['summary']['total_messages'] += channel_data_item.get('messages', 0)
            dashboard_data['summary']['total_files'] += channel_data_item.get('files', 0)
        
        return dashboard_data

    def _create_empty_dashboard_data(self) -> Dict[str, Any]:
        """Create empty dashboard data structure."""
        return {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_channels': 0,
                'total_files': 0
            },
            'summary': {
                'total_messages': 0,
                'total_files': 0,
                'channels_count': 0
            },
            'channels': {}
        }

    # Template Rendering Methods
    def _render_index_page(self, data: Dict[str, Any]) -> str:
        """Generate main dashboard page."""
        try:
            template = self.template_env.get_template('index.html')
            return template.render(
                title="Telegram Channel Analysis Dashboard",
                data=data,
                ga_measurement_id=DASHBOARD_GA_MEASUREMENT_ID if DASHBOARD_GA_ENABLED else None
            )
        except Exception as e:
            self.logger.error(f"Error rendering index page: {e}")
            raise

    def _render_channel_page(self, channel_name: str, data: Dict[str, Any]) -> str:
        """Generate individual channel page."""
        try:
            template = self.template_env.get_template('channel.html')
            channel_data = data['channels'].get(channel_name, {})
            
            return template.render(
                title=f"{channel_name} - Channel Analysis",
                channel_name=channel_name,
                channel_data=channel_data,
                data=data,
                ga_measurement_id=DASHBOARD_GA_MEASUREMENT_ID if DASHBOARD_GA_ENABLED else None
            )
        except Exception as e:
            self.logger.error(f"Error rendering channel page for {channel_name}: {e}")
            raise

    def _render_shared_css(self) -> str:
        """Generate CSS styles."""
        return """
/* Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
}

.header {
    background: #fff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 1rem 0;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: bold;
    color: #007bff;
}

.nav {
    display: flex;
    gap: 2rem;
}

.nav-item {
    text-decoration: none;
    color: #666;
    font-weight: 500;
}

.nav-item.active {
    color: #007bff;
}

.btn {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    font-weight: 500;
}

.btn-primary {
    background: #007bff;
    color: white;
}

.btn-secondary {
    background: #6c757d;
    color: white;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #007bff;
}

.metric-label {
    color: #666;
    margin-top: 0.5rem;
}

.channels-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.channel-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.channel-name {
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

.channel-stats {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}

.channel-stat-value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #28a745;
}

.channel-stat-label {
    color: #666;
    font-size: 0.9rem;
}

.chart-section {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 2rem 0;
}

.chart-container {
    height: 300px !important;
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
    display: block;
}

/* Force Chart.js to respect container height */
.chart-container canvas {
    height: 300px !important;
    width: 100% !important;
}

@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        gap: 1rem;
    }
    
    .nav {
        gap: 1rem;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .channels-grid {
        grid-template-columns: 1fr;
    }
}
"""

    def _render_shared_js(self) -> str:
        """Generate JavaScript with Google Analytics."""
        return f"""
// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {{
    initializeCharts();
    initializeAnalytics();
}});

function initializeCharts() {{
    // Get chart data from script tag
    const chartDataScript = document.getElementById('chart-data');
    if (!chartDataScript) {{
        console.error('Chart data script not found');
        return;
    }}
    
    let chartData;
    try {{
        chartData = JSON.parse(chartDataScript.textContent);
    }} catch (error) {{
        console.error('Error parsing chart data:', error);
        return;
    }}
    
    // Initialize Chart.js charts
    const chartElements = document.querySelectorAll('.chart-container');
    
    chartElements.forEach((element, index) => {{
        const chartType = element.dataset.chartType;
        console.log('Processing chart element:', chartType, 'element:', element);
        
        if (chartType && chartData[chartType]) {{
            try {{
                const chartConfig = chartData[chartType];
                console.log('Chart config for', chartType, ':', chartConfig);
                new Chart(element, chartConfig);
                console.log('Chart initialized successfully:', chartType);
            }} catch (error) {{
                console.error('Error creating chart for', chartType, ':', error);
            }}
        }} else {{
            console.warn('No chart data found for type:', chartType, 'Available types:', Object.keys(chartData));
        }}
    }});
}}

function initializeAnalytics() {{
    // Google Analytics initialization
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'dashboard_loaded', {{
            'event_category': 'dashboard',
            'event_label': 'index_page'
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
}}

function handleExport(type) {{
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'export_action', {{
            'event_category': 'dashboard',
            'event_label': type
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
    
    // Export functionality
    console.log('Export requested:', type);
}}

function handleChartInteraction(chartType, action, channelName) {{
    {f'''
    if (typeof gtag !== 'undefined') {{
        gtag('event', 'chart_interaction', {{
            'event_category': 'dashboard',
            'event_label': `${{chartType}}_${{action}}_${{channelName}}`
        }});
    }}
    ''' if DASHBOARD_GA_ENABLED else ''}
}}
"""

    # File Generation Methods
    def _generate_shared_files(self):
        """Create CSS, JS, and data files."""
        try:
            # Generate CSS
            css_content = self._render_shared_css()
            css_file = self.output_dir / DASHBOARD_CSS_PATH / DASHBOARD_CSS_FILENAME
            css_file.write_text(css_content, encoding='utf-8')
            self.logger.debug(f"Generated CSS file: {css_file}")
            
            # Generate JavaScript
            js_content = self._render_shared_js()
            js_file = self.output_dir / DASHBOARD_JS_PATH / "dashboard_new.js"
            js_file.write_text(js_content, encoding='utf-8')
            self.logger.debug(f"Generated JavaScript file: {js_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating shared files: {e}")
            raise

    def _generate_html_pages(self, data: Dict[str, Any]):
        """Create all HTML pages."""
        try:
            # Generate index page
            index_content = self._render_index_page(data)
            index_file = self.output_dir / DASHBOARD_INDEX_FILENAME
            index_file.write_text(index_content, encoding='utf-8')
            self.logger.debug(f"Generated index page: {index_file}")
            
            # Generate channel pages
            for channel_name in data['channels'].keys():
                try:
                    channel_content = self._render_channel_page(channel_name, data)
                    sanitized_name = self._sanitize_channel_name(channel_name)
                    channel_file = self.output_dir / f"{sanitized_name}.html"
                    channel_file.write_text(channel_content, encoding='utf-8')
                    self.logger.debug(f"Generated channel page: {channel_file}")
                except Exception as e:
                    self.logger.error(f"Error generating page for channel {channel_name}: {e}")
                    continue
            
            # Generate data file
            data_file = self.output_dir / 'data' / DASHBOARD_DATA_FILENAME
            data_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
            self.logger.debug(f"Generated data file: {data_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating HTML pages: {e}")
            raise
    
    def _extract_filename_metrics(self, filename_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed filename metrics for display."""
        duplicate_detection = filename_data.get('duplicate_filename_detection', {})
        pattern_analysis = filename_data.get('filename_pattern_analysis', {})
        
        return {
            'duplicate_ratio': duplicate_detection.get('duplicate_ratio', 0),
            'files_with_duplicate_names': duplicate_detection.get('files_with_duplicate_names', 0),
            'total_unique_filenames': duplicate_detection.get('total_unique_filenames', 0),
            'total_files': duplicate_detection.get('total_files', 0),
            'most_common_filenames': duplicate_detection.get('most_common_filenames', [])[:10],  # Top 10
            'files_with_special_chars': pattern_analysis.get('files_with_special_chars', 0),
            'files_with_spaces': pattern_analysis.get('files_with_spaces', 0),
            'filename_length_stats': pattern_analysis.get('filename_length', {})
        }
    
    def _extract_filesize_metrics(self, filesize_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed filesize metrics for display."""
        duplicate_detection = filesize_data.get('duplicate_filesize_detection', {})
        distribution_analysis = filesize_data.get('filesize_distribution_analysis', {})
        
        return {
            'duplicate_ratio': duplicate_detection.get('duplicate_ratio', 0),
            'files_with_duplicate_sizes': duplicate_detection.get('files_with_duplicate_sizes', 0),
            'total_unique_filesizes': duplicate_detection.get('total_unique_filesizes', 0),
            'total_files': duplicate_detection.get('total_files', 0),
            'most_common_filesizes': duplicate_detection.get('most_common_filesizes', [])[:10],  # Top 10
            'size_frequency_distribution': distribution_analysis.get('size_frequency_distribution', {}),
            'potential_duplicates_by_size': distribution_analysis.get('potential_duplicates_by_size', [])[:10]
        }
    
    def _extract_message_metrics(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed message metrics for display."""
        content_stats = message_data.get('content_statistics', {})
        pattern_recognition = message_data.get('pattern_recognition', {})
        language_detection = message_data.get('language_detection', {})
        
        return {
            'total_messages': content_stats.get('total_messages', 0),
            'messages_with_text': content_stats.get('messages_with_text', 0),
            'media_messages': content_stats.get('media_messages', 0),
            'forwarded_messages': content_stats.get('forwarded_messages', 0),
            'text_length_stats': content_stats.get('text_length_stats', {}),
            'hashtags_count': pattern_recognition.get('hashtags', {}).get('total_unique_hashtags', 0),
            'mentions_count': pattern_recognition.get('mentions', {}).get('total_unique_mentions', 0),
            'urls_count': pattern_recognition.get('urls', {}).get('total_unique_urls', 0),
            'emojis_count': pattern_recognition.get('emojis', {}).get('total_unique_emojis', 0),
            'detected_languages': language_detection.get('detected_languages', []),
            'primary_language': language_detection.get('primary_language', 'Unknown'),
            'language_confidence': language_detection.get('confidence', 0)
        }
    
    def _transform_mentions_analysis(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform mentions analysis data for charts."""
        mentions_data = message_data.get('pattern_recognition', {}).get('mentions', {})
        top_mentions = mentions_data.get('top_mentions', [])[:15]  # Top 15
        
        if not top_mentions:
            return None
        
        return {
            'type': 'bar',
            'data': {
                'labels': [mention['username'] for mention in top_mentions],
                'datasets': [{
                    'label': 'Mention Count',
                    'data': [mention['count'] for mention in top_mentions],
                    'backgroundColor': '#17a2b8',
                    'borderColor': '#138496',
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Top Mentions'
                    },
                    'legend': {
                        'display': False
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    },
                    'x': {
                        'ticks': {
                            'maxRotation': 45
                        }
                    }
                }
            }
        }
    
    def _transform_urls_analysis(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform URLs analysis data for charts."""
        urls_data = message_data.get('pattern_recognition', {}).get('urls', {})
        top_urls = urls_data.get('top_urls', [])[:15]  # Top 15
        
        if not top_urls:
            return None
        
        return {
            'type': 'bar',
            'data': {
                'labels': [url['url'] for url in top_urls],
                'datasets': [{
                    'label': 'URL Count',
                    'data': [url['count'] for url in top_urls],
                    'backgroundColor': '#28a745',
                    'borderColor': '#1e7e34',
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Top URLs'
                    },
                    'legend': {
                        'display': False
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    },
                    'x': {
                        'ticks': {
                            'maxRotation': 45
                        }
                    }
                }
            }
        }
    
    def _transform_language_analysis(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform language detection data for charts."""
        language_data = message_data.get('language_detection', {})
        detected_languages = language_data.get('detected_languages', [])
        
        if not detected_languages:
            return None
        
        # Take top 10 languages
        top_languages = detected_languages[:10]
        
        return {
            'type': 'doughnut',
            'data': {
                'labels': [lang['language'] for lang in top_languages],
                'datasets': [{
                    'label': 'Message Count',
                    'data': [lang['count'] for lang in top_languages],
                    'backgroundColor': [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                        '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#FF6384'
                    ],
                    'borderWidth': 2
                }]
            },
            'options': {
                'responsive': True,
                'maintainAspectRatio': False,
                'aspectRatio': 1,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Language Distribution'
                    },
                    'legend': {
                        'position': 'bottom'
                    }
                }
            }
        }
    def _extract_creator_metrics(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed creator metrics for display."""
        creator_data = message_data.get('creator_analysis', {})
        
        return {
            'total_creators': creator_data.get('total_creators', 0),
            'most_active_creators': creator_data.get('most_active_creators', [])[:10],  # Top 10
            'creator_message_stats': creator_data.get('creator_message_stats', {}),
            'total_messages': creator_data.get('creator_message_stats', {}).get('total_messages', 0),
            'avg_messages_per_creator': creator_data.get('creator_message_stats', {}).get('avg_messages_per_creator', 0)
        }
    
    def _transform_creator_analysis(self, message_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform creator analysis data for charts."""
        creator_data = message_data.get('creator_analysis', {})
        most_active_creators = creator_data.get('most_active_creators', [])
        
        if not most_active_creators:
            return None
        
        # Take top 15 creators
        top_creators = most_active_creators[:15]
        
        return {
            'type': 'bar',
            'data': {
                'labels': [creator['username'] for creator in top_creators],
                'datasets': [{
                    'label': 'Message Count',
                    'data': [creator['message_count'] for creator in top_creators],
                    'backgroundColor': '#6f42c1',
                    'borderColor': '#5a32a3',
                    'borderWidth': 1
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Top Contributors by Message Count'
                    },
                    'legend': {
                        'display': False
                    }
                },
                'scales': {
                    'y': {
                        'beginAtZero': True
                    },
                    'x': {
                        'ticks': {
                            'maxRotation': 45
                        }
                    }
                }
            }
        }
