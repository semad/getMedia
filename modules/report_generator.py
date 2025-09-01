"""
Report Generator Module

This module contains the core reporting functions for generating comprehensive
analysis reports from Telegram message data.
"""

import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_pandas_report(df: pd.DataFrame, channel_name: str) -> dict[str, Any]:
    """
    Generate a comprehensive report using pandas DataFrame with detailed field analysis.

    Args:
        df: Pandas DataFrame containing message data
        channel_name: Name of the channel being analyzed

    Returns:
        Dictionary containing comprehensive analysis report
    """
    try:
        # Basic statistics
        total_messages = len(df)

        # Field analysis - comprehensive analysis of each column
        field_analysis = _analyze_fields(df)

        # Media analysis
        media_stats = _analyze_media(df)

        # Date analysis
        date_stats = _analyze_dates(df)

        # File size analysis
        file_stats = _analyze_file_sizes(df)

        # Engagement analysis
        engagement_stats = _analyze_engagement(df)

        # Create comprehensive report
        report = {
            "channel_name": channel_name,
            "generated_at": datetime.now().isoformat(),
            "total_messages": total_messages,
            "total_columns": len(df.columns),
            "dataframe_shape": f"{df.shape[0]} rows x {df.shape[1]} columns",
            "media_messages": media_stats["media_count"],
            "text_messages": media_stats["text_count"],
            "total_file_size": file_stats["total_size"],
            "date_range": date_stats["date_range"],
            "active_days": date_stats["active_days"],
            "field_analysis": field_analysis,
            "media_analysis": media_stats,
            "date_analysis": date_stats,
            "file_analysis": file_stats,
            "engagement_analysis": engagement_stats,
        }

        return report

    except Exception as e:
        logger.error(f"Error generating report for {channel_name}: {e}")
        # Return minimal report on error
        return {
            "channel_name": channel_name,
            "error": str(e),
            "total_messages": len(df) if df is not None else 0,
        }


def _analyze_fields(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze all fields in the DataFrame."""
    field_analysis = {}

    for column in df.columns:
        col_data = df[column]
        field_info = {
            "data_type": str(col_data.dtype),
            "total_values": len(col_data),
            "non_null_count": col_data.count(),
            "null_count": col_data.isnull().sum(),
            "null_percentage": round(
                (col_data.isnull().sum() / len(col_data)) * 100, 2
            ),
            "unique_count": col_data.nunique(),
            "unique_percentage": round((col_data.nunique() / len(col_data)) * 100, 2),
        }

        # Add specific analysis based on data type
        if col_data.dtype in ["int64", "float64"]:
            field_info.update(_analyze_numeric_field(col_data))
        elif col_data.dtype == "object":
            field_info.update(_analyze_text_field(col_data))

        field_analysis[column] = field_info

    return field_analysis


def _analyze_numeric_field(col_data: pd.Series) -> dict[str, Any]:
    """Analyze numeric fields with statistical measures."""
    if col_data.empty:
        return {
            "min_value": None,
            "max_value": None,
            "mean_value": None,
            "median_value": None,
            "std_deviation": None,
        }

    return {
        "min_value": float(col_data.min()),
        "max_value": float(col_data.max()),
        "mean_value": float(col_data.mean()),
        "median_value": float(col_data.median()),
        "std_deviation": float(col_data.std()),
    }


def _analyze_text_field(col_data: pd.Series) -> dict[str, Any]:
    """Analyze text fields with string statistics."""
    non_null_text = col_data.dropna()

    if len(non_null_text) == 0:
        return {
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "empty_strings": 0,
            "top_values": {},
        }

    # Handle lists and complex objects safely
    try:
        top_values = col_data.value_counts().head(5).to_dict()
    except (TypeError, AttributeError):
        top_values = "Complex data type - cannot count values"

    return {
        "avg_length": round(non_null_text.astype(str).str.len().mean(), 2),
        "max_length": int(non_null_text.astype(str).str.len().max()),
        "min_length": int(non_null_text.astype(str).str.len().min()),
        "empty_strings": int((non_null_text == "").sum()),
        "top_values": top_values,
    }


def _analyze_media(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze media-related fields."""
    if "media_type" in df.columns:
        media_messages = len(df[df["media_type"].notna() & (df["media_type"] != "")])
        try:
            media_types = df["media_type"].value_counts().to_dict()
        except (TypeError, AttributeError):
            media_types = "Complex data type - cannot count values"
    else:
        media_messages = 0
        media_types = {}

    text_messages = len(df) - media_messages

    return {
        "media_count": media_messages,
        "text_count": text_messages,
        "media_types": media_types,
        "media_percentage": round((media_messages / len(df)) * 100, 2)
        if len(df) > 0
        else 0,
    }


def _analyze_dates(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze date-related fields."""
    if "date" in df.columns:
        try:
            # Convert to datetime if possible
            date_col = pd.to_datetime(df["date"], errors="coerce")
            valid_dates = date_col.dropna()

            if len(valid_dates) > 0:
                date_range = f"{valid_dates.min()} to {valid_dates.max()}"
                active_days = valid_dates.dt.date.nunique()
            else:
                date_range = "No valid dates found"
                active_days = 0
        except Exception:
            date_range = "Date parsing failed"
            active_days = 0
    else:
        date_range = "No date column found"
        active_days = 0

    return {
        "date_range": date_range,
        "active_days": active_days,
    }


def _analyze_file_sizes(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze file size information."""
    if "file_size" in df.columns:
        try:
            file_sizes = pd.to_numeric(df["file_size"], errors="coerce")
            valid_sizes = file_sizes.dropna()

            if len(valid_sizes) > 0:
                total_size = valid_sizes.sum()
                avg_size = valid_sizes.mean()
                max_size = valid_sizes.max()
                min_size = valid_sizes.min()
            else:
                total_size = avg_size = max_size = min_size = 0
        except Exception:
            total_size = avg_size = max_size = min_size = 0
    else:
        total_size = avg_size = max_size = min_size = 0

    return {
        "total_size": total_size,
        "average_size": avg_size,
        "max_size": max_size,
        "min_size": min_size,
        "size_unit": "bytes",
    }


def _analyze_engagement(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze engagement metrics."""
    engagement_stats = {}

    # Views analysis
    if "views" in df.columns:
        try:
            views = pd.to_numeric(df["views"], errors="coerce")
            valid_views = views.dropna()
            if len(valid_views) > 0:
                engagement_stats["total_views"] = int(valid_views.sum())
                engagement_stats["avg_views"] = round(valid_views.mean(), 2)
                engagement_stats["max_views"] = int(valid_views.max())
        except Exception:
            engagement_stats["views_error"] = "Could not analyze views"

    # Forwards analysis
    if "forwards" in df.columns:
        try:
            forwards = pd.to_numeric(df["forwards"], errors="coerce")
            valid_forwards = forwards.dropna()
            if len(valid_forwards) > 0:
                engagement_stats["total_forwards"] = int(valid_forwards.sum())
                engagement_stats["avg_forwards"] = round(valid_forwards.mean(), 2)
                engagement_stats["max_forwards"] = int(valid_forwards.max())
        except Exception:
            engagement_stats["forwards_error"] = "Could not analyze forwards"

    # Replies analysis
    if "replies" in df.columns:
        try:
            replies = pd.to_numeric(df["replies"], errors="coerce")
            valid_replies = replies.dropna()
            if len(valid_replies) > 0:
                engagement_stats["total_replies"] = int(valid_replies.sum())
                engagement_stats["avg_replies"] = round(valid_replies.mean(), 2)
                engagement_stats["max_replies"] = int(valid_replies.max())
        except Exception:
            engagement_stats["replies_error"] = "Could not analyze replies"

    return engagement_stats


def generate_field_discovery_report(df, channel_name: str) -> dict:
    """Generate a comprehensive field discovery report for the dataset."""
    try:
        field_analysis = {}

        for column in df.columns:
            col_data = df[column]

            # Basic field info
            field_info = {
                "data_type": str(col_data.dtype),
                "total_values": len(col_data),
                "non_null_values": col_data.count(),
                "null_values": col_data.isnull().sum(),
                "null_percentage": round(
                    (col_data.isnull().sum() / len(col_data)) * 100, 2
                ),
                "unique_values": col_data.nunique(),
                "unique_percentage": round(
                    (col_data.nunique() / len(col_data)) * 100, 2
                ),
            }

            # Value distribution analysis
            if col_data.dtype in ["int64", "float64"]:
                if not col_data.empty:
                    field_info.update(
                        {
                            "min_value": float(col_data.min()),
                            "max_value": float(col_data.max()),
                            "mean_value": float(col_data.mean()),
                            "median_value": float(col_data.median()),
                            "std_deviation": float(col_data.std()),
                            "value_range": f"{col_data.min()} to {col_data.max()}",
                        }
                    )

            elif col_data.dtype == "object":
                non_null_text = col_data.dropna()
                if len(non_null_text) > 0:
                    # Handle lists and complex objects safely
                    try:
                        # Try to get value counts, but handle lists safely
                        value_counts = col_data.value_counts()
                        top_values = value_counts.head(10).to_dict()
                    except TypeError:
                        # If we can't get value counts (e.g., due to lists), use a safe approach
                        top_values = "Complex data type - cannot count values"

                    field_info.update(
                        {
                            "avg_length": round(
                                non_null_text.astype(str).str.len().mean(), 2
                            ),
                            "max_length": int(
                                non_null_text.astype(str).str.len().max()
                            ),
                            "min_length": int(
                                non_null_text.astype(str).str.len().min()
                            ),
                            "empty_strings": int((non_null_text == "").sum()),
                            "top_values": top_values,
                            "sample_values": non_null_text.head(5).tolist(),
                        }
                    )

            elif col_data.dtype == "bool":
                if not col_data.empty:
                    try:
                        value_counts = col_data.value_counts()
                        field_info.update(
                            {
                                "true_count": int(value_counts.get(True, 0)),
                                "false_count": int(value_counts.get(False, 0)),
                                "true_percentage": round(
                                    (value_counts.get(True, 0) / len(col_data)) * 100, 2
                                ),
                            }
                        )
                    except TypeError:
                        field_info.update(
                            {"true_count": 0, "false_count": 0, "true_percentage": 0.0}
                        )

            field_analysis[column] = field_info

        # Dataset-level analysis
        dataset_analysis = {
            "total_fields": len(df.columns),
            "field_types": df.dtypes.value_counts().to_dict(),
            "completeness_score": round(
                (df.count().sum() / (len(df) * len(df.columns))) * 100, 2
            ),
            "field_categories": {
                "numeric_fields": [
                    col for col in df.columns if df[col].dtype in ["int64", "float64"]
                ],
                "text_fields": [col for col in df.columns if df[col].dtype == "object"],
                "boolean_fields": [
                    col for col in df.columns if df[col].dtype == "bool"
                ],
                "datetime_fields": [
                    col
                    for col in df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ],
            },
        }

        return {
            "channel_name": channel_name,
            "generated_at": datetime.now().isoformat(),
            "field_analysis": field_analysis,
            "dataset_analysis": dataset_analysis,
            "report_type": "field_discovery_analysis",
        }

    except Exception as e:
        return {
            "channel_name": channel_name,
            "generated_at": datetime.now().isoformat(),
            "error": str(e),
            "report_type": "field_discovery_analysis_error",
        }


async def generate_channel_overview_report(db_service) -> dict:
    """Generate a channel overview report from database statistics."""
    try:
        async with db_service:
            # Get basic stats which include channel information
            stats = await db_service.get_stats()

            # Create channel overview report
            channel_report = {
                "timestamp": datetime.now().isoformat(),
                "report_type": "channel_overview",
                "generated_by": "reporting.report_generator",
                "summary": {
                    "total_messages": stats.get("total_messages", 0),
                    "total_channels": stats.get("total_channels", 0),
                    "media_messages": stats.get("media_messages", 0),
                    "text_messages": stats.get("text_messages", 0),
                },
            }

            return channel_report
    except Exception as e:
        logger.error(f"Error generating channel overview report: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "report_type": "channel_overview_error",
            "error": str(e),
        }
