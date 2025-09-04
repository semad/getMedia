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
        # Create a pie chart for file uniqueness
        unique_files = data.get('unique_filenames', 0)
        duplicate_files = data.get('duplicate_filenames', 0)
        
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
        # Create a bar chart for file size distribution
        size_distribution = data.get('size_distribution', {})
        
        labels = []
        values = []
        colors = []
        
        for size_category, count in size_distribution.items():
            labels.append(size_category.title())
            values.append(count)
            # Color coding based on size category
            if size_category == 'tiny':
                colors.append('#17a2b8')
            elif size_category == 'small':
                colors.append('#28a745')
            elif size_category == 'medium':
                colors.append('#ffc107')
            elif size_category == 'large':
                colors.append('#fd7e14')
            elif size_category == 'huge':
                colors.append('#dc3545')
            else:
                colors.append('#6c757d')
        
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
    
    def _transform_message_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform message analysis data for charts."""
        # Create a line chart for temporal patterns (hourly)
        temporal_patterns = data.get('temporal_patterns', {})
        hourly_data = temporal_patterns.get('hourly', {})
        
        hours = []
        counts = []
        
        for hour in range(24):
            hours.append(f"{hour:02d}:00")
            counts.append(hourly_data.get(str(hour), 0))
        
        return {
            'type': 'line',
            'data': {
                'labels': hours,
                'datasets': [{
                    'label': 'Messages per Hour',
                    'data': counts,
                    'borderColor': '#007bff',
                    'backgroundColor': 'rgba(0, 123, 255, 0.1)',
                    'borderWidth': 2,
                    'fill': True,
                    'tension': 0.4
                }]
            },
            'options': {
                'responsive': True,
                'plugins': {
                    'title': {
                        'display': True,
                        'text': 'Message Activity by Hour'
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
            collections_dir = Path("reports/collections")
            collection_files = list(collections_dir.glob(f"*{channel_name}*combined.json"))
            
            if not collection_files:
                self.logger.warning(f"Collection file not found for {channel_name}")
                return None
            
            collection_file = collection_files[0]
            
            with open(collection_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data or len(data) == 0:
                return None
            
            # Extract messages from the collection data
            messages = data[0].get('messages', [])
            if not messages:
                return None
            
            # Process dates and create timeline data
            from datetime import datetime
            import pandas as pd
            
            # Extract dates and sort them
            dates = []
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
                            dates.append(date_obj)
                    except Exception as e:
                        self.logger.debug(f"Error parsing date {date_str}: {e}")
                        continue
            
            if not dates:
                return None
            
            # Sort dates
            dates.sort()
            
            # Create timeline data (daily aggregation)
            timeline_data = {}
            for date in dates:
                date_key = date.strftime('%Y-%m-%d')
                timeline_data[date_key] = timeline_data.get(date_key, 0) + 1
            
            # Convert to chart format
            return self._transform_timeline_data(timeline_data)
            
        except Exception as e:
            self.logger.error(f"Error getting timeline data for {channel_name}: {e}")
            return None

    def _get_monthly_histogram_for_channel(self, channel_name: str) -> Optional[Dict[str, Any]]:
        """Get monthly histogram data for a specific channel from diff_messages source."""
        try:
            # Use diff_messages as the primary source for monthly data
            message_analysis_file = Path(self.input_dir) / "diff_messages" / channel_name / "message_analysis.json"
            
            if not message_analysis_file.exists():
                self.logger.warning(f"Message analysis file not found for {channel_name}")
                return None
            
            with open(message_analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data or len(data) == 0:
                return None
            
            # The data is a list, get the first item
            message_data = data[0] if isinstance(data, list) else data
            
            # Transform the monthly data
            return self._transform_monthly_histogram(message_data)
            
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
        
        for source_type in DASHBOARD_SUPPORTED_SOURCE_TYPES:
            source_path = input_path / source_type
            if source_path.exists():
                files[source_type] = list(source_path.rglob("*.json"))
                self.logger.debug(f"Found {len(files[source_type])} files in {source_type}")
        
        return files

    def _load_analysis_data(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and validate JSON data from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            if not isinstance(data, list) or not data:
                self.logger.warning(f"Invalid data format in {file_path}")
                return None
            
            for item in data:
                if not all(key in item for key in ['report_type', 'generated_at']):
                    self.logger.warning(f"Missing required fields in {file_path}")
                    return None
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

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
                # Validate filename analysis has basic fields
                if not any(field in data for field in ['total_files', 'unique_filenames', 'duplicate_filenames']):
                    self.logger.warning(f"Missing required fields in {report_type}")
                    return False
            elif report_type == 'filesize_analysis':
                # Validate filesize analysis has basic fields
                if not any(field in data for field in ['total_size_bytes', 'total_size_mb', 'avg_file_size_mb']):
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
                
                # Process each report in the data
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
    height: 300px;
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
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
        
        if (chartType && chartData[chartType]) {{
            try {{
                const chartConfig = chartData[chartType];
                new Chart(element, chartConfig);
                console.log('Chart initialized:', chartType);
            }} catch (error) {{
                console.error('Error creating chart for', chartType, ':', error);
            }}
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
