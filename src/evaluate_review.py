"""
Evaluate human reviews from Argilla and generate reports.
"""

import os
import argparse
import argilla as rg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import logging
from argilla import ApiClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate human reviews from Argilla"
    )
    parser.add_argument(
        "--dataset-name", type=str, default="doctriage_review",
        help="Name of the Argilla dataset to evaluate"
    )
    parser.add_argument(
        "--argilla-api-url", type=str, default="http://localhost:6900",
        help="Argilla API URL"
    )
    parser.add_argument(
        "--workspace", type=str, default="admin",
        help="Argilla workspace name"
    )
    parser.add_argument(
        "--api-key", type=str, default="admin.apikey",
        help="Argilla API key"
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--feedback-status", type=str, default="submitted",
        help="Filter records by feedback status (e.g., 'submitted', 'discarded', 'draft')"
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export annotated data for retraining"
    )
    parser.add_argument(
        "--export-dir", type=str, default="data/feedback",
        help="Directory to save exported data for retraining"
    )
    return parser.parse_args()

def init_argilla(args):
    """Initialize Argilla client."""
    # Configure Argilla client
    api_client = ApiClient(
        api_url=args.argilla_api_url,
        api_key=args.api_key,
        workspace=args.workspace
    )
    
    # Get dataset
    try:
        dataset = rg.FeedbackDataset.from_argilla(
            name=args.dataset_name,
            api_client=api_client
        )
        logger.info(f"Loaded dataset: {args.dataset_name} with {len(dataset)} records")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise RuntimeError(f"Failed to load dataset: {str(e)}")

def load_records(dataset, args):
    """Load records from Argilla dataset with specified feedback status."""
    # Convert to DataFrame
    df = dataset.to_pandas()
    
    # Filter by feedback status if specified
    if args.feedback_status:
        df = df[df["status"] == args.feedback_status]
        logger.info(f"Filtered to {len(df)} records with status '{args.feedback_status}'")
    
    return df

def analyze_corrections(df):
    """Analyze human corrections to model predictions."""
    # Create column for whether prediction was corrected
    df["corrected"] = df["prediction"] != df["annotation"]
    
    # Count corrections by predicted class
    corrections_by_class = df.groupby("prediction")["corrected"].mean().reset_index()
    corrections_by_class.columns = ["predicted_class", "correction_rate"]
    
    # Count how predictions were corrected
    correction_flows = df[df["corrected"]].groupby(["prediction", "annotation"]).size().reset_index()
    correction_flows.columns = ["from", "to", "count"]
    
    # Calculate overall correction rate
    overall_correction_rate = df["corrected"].mean()
    
    # Calculate agreement with original labels vs. model predictions
    if "metadata" in df.columns and df["metadata"].apply(lambda x: "original_label" in x).any():
        df["original_label"] = df["metadata"].apply(lambda x: x.get("original_label", ""))
        agreement_with_original = (df["annotation"] == df["original_label"]).mean()
        agreement_with_model = (df["annotation"] == df["prediction"]).mean()
    else:
        agreement_with_original = None
        agreement_with_model = 1 - overall_correction_rate
    
    return {
        "overall_correction_rate": overall_correction_rate,
        "corrections_by_class": corrections_by_class,
        "correction_flows": correction_flows,
        "agreement_with_original": agreement_with_original,
        "agreement_with_model": agreement_with_model
    }

def analyze_confidence(df):
    """Analyze relationship between model confidence and corrections."""
    # Extract confidence from metadata
    df["confidence"] = df["metadata"].apply(lambda x: x.get("confidence", 0))
    
    # Group records into confidence bins
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    df["confidence_bin"] = pd.cut(df["confidence"], bins)
    
    # Calculate correction rate by confidence bin
    correction_by_confidence = df.groupby("confidence_bin")["corrected"].agg(
        ["mean", "count"]
    ).reset_index()
    correction_by_confidence.columns = ["confidence_bin", "correction_rate", "count"]
    
    # Calculate metrics
    confidence_corrected_mean = df[df["corrected"]]["confidence"].mean()
    confidence_not_corrected_mean = df[~df["corrected"]]["confidence"].mean()
    
    return {
        "correction_by_confidence": correction_by_confidence,
        "confidence_corrected_mean": confidence_corrected_mean,
        "confidence_not_corrected_mean": confidence_not_corrected_mean
    }

def generate_classification_metrics(df):
    """Generate classification metrics based on human annotations vs. model predictions."""
    # Extract predictions and annotations
    y_pred = df["prediction"].values
    y_true = df["annotation"].values
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=["reports", "regulations"])
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred, labels=["reports", "regulations"], output_dict=True)
    
    return {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report
    }

def plot_confidence_vs_corrections(confidence_analysis, output_path):
    """Plot relationship between model confidence and correction rate."""
    correction_by_conf = confidence_analysis["correction_by_confidence"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot correction rate by confidence bin
    ax = sns.barplot(x=correction_by_conf["confidence_bin"].astype(str), 
                    y=correction_by_conf["correction_rate"])
    
    # Add count annotations
    for i, row in enumerate(correction_by_conf.itertuples()):
        ax.text(i, row.correction_rate + 0.01, f"n={row.count}", 
                ha='center', va='bottom', fontsize=9)
    
    plt.title("Correction Rate by Model Confidence")
    plt.xlabel("Model Confidence")
    plt.ylabel("Correction Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(conf_matrix, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
               xticklabels=["reports", "regulations"],
               yticklabels=["reports", "regulations"])
    
    plt.title("Confusion Matrix (Human Annotations vs. Model Predictions)")
    plt.xlabel("Predicted Label")
    plt.ylabel("Human Annotation")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_correction_flow_plot(correction_analysis, output_path):
    """Generate plot showing how predictions were corrected."""
    correction_flows = correction_analysis["correction_flows"]
    
    plt.figure(figsize=(8, 6))
    
    # Create a directional graph-like visualization
    for i, row in enumerate(correction_flows.itertuples()):
        plt.arrow(0, i, 1, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
        plt.text(-0.1, i, f"{row._1}", ha="right", va="center", fontsize=12)
        plt.text(1.1, i, f"{row._2}", ha="left", va="center", fontsize=12)
        plt.text(0.5, i, f"n={row._3}", ha="center", va="center", fontsize=10)
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, len(correction_flows) - 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.title("Correction Flows")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(analyses, args):
    """Generate evaluation report with tables and plots."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    plot_confidence_vs_corrections(
        analyses["confidence_analysis"],
        os.path.join(args.output_dir, "confidence_vs_corrections.png")
    )
    
    plot_confusion_matrix(
        analyses["classification_metrics"]["confusion_matrix"],
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    if not analyses["correction_analysis"]["correction_flows"].empty:
        generate_correction_flow_plot(
            analyses["correction_analysis"],
            os.path.join(args.output_dir, "correction_flows.png")
        )
    
    # Generate HTML report
    corrections_by_class = analyses["correction_analysis"]["corrections_by_class"]
    classification_report_df = pd.DataFrame(analyses["classification_metrics"]["classification_report"]).T
    confidence_by_bin = analyses["confidence_analysis"]["correction_by_confidence"]
    
    html_report = f"""
    <html>
    <head>
        <title>DocTriage Human Review Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .summary {{ background-color: #eef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>DocTriage-BERT Human Review Evaluation</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><b>Dataset:</b> {args.dataset_name}</p>
            <p><b>Overall Correction Rate:</b> {analyses["correction_analysis"]["overall_correction_rate"]:.2%}</p>
            <p><b>Agreement with Model:</b> {analyses["correction_analysis"]["agreement_with_model"]:.2%}</p>
    """
    
    if analyses["correction_analysis"]["agreement_with_original"] is not None:
        html_report += f"""
            <p><b>Agreement with Original Labels:</b> {analyses["correction_analysis"]["agreement_with_original"]:.2%}</p>
        """
    
    html_report += f"""
        </div>
        
        <h2>Corrections by Model Confidence</h2>
        <p>Mean confidence when model was corrected: {analyses["confidence_analysis"]["confidence_corrected_mean"]:.4f}</p>
        <p>Mean confidence when model was not corrected: {analyses["confidence_analysis"]["confidence_not_corrected_mean"]:.4f}</p>
        <img src="confidence_vs_corrections.png" alt="Confidence vs Corrections">
        
        <h2>Confusion Matrix</h2>
        <img src="confusion_matrix.png" alt="Confusion Matrix">
        
        <h2>Correction Flows</h2>
    """
    
    if not analyses["correction_analysis"]["correction_flows"].empty:
        html_report += f"""
        <img src="correction_flows.png" alt="Correction Flows">
        """
    else:
        html_report += f"""
        <p>No corrections were made.</p>
        """
    
    html_report += f"""
        <h2>Corrections by Predicted Class</h2>
        {corrections_by_class.to_html(index=False)}
        
        <h2>Classification Report</h2>
        {classification_report_df.to_html()}
        
        <h2>Confidence Bins</h2>
        {confidence_by_bin.to_html(index=False)}
        
    </body>
    </html>
    """
    
    # Write HTML report
    with open(os.path.join(args.output_dir, "evaluation_report.html"), "w") as f:
        f.write(html_report)
    
    # Write CSV files
    corrections_by_class.to_csv(os.path.join(args.output_dir, "corrections_by_class.csv"), index=False)
    classification_report_df.to_csv(os.path.join(args.output_dir, "classification_report.csv"))
    confidence_by_bin.to_csv(os.path.join(args.output_dir, "confidence_bins.csv"), index=False)
    
    logger.info(f"Report generated in {args.output_dir}")
    logger.info(f"Open {os.path.join(args.output_dir, 'evaluation_report.html')} to view the results")

def export_for_retraining(df, args):
    """Export annotated data for retraining the model."""
    # Create export directory
    os.makedirs(args.export_dir, exist_ok=True)
    
    # Prepare data for retraining - using human annotations as ground truth
    export_data = df[["text", "annotation"]].copy()
    export_data.columns = ["text", "domain"]  # Match expected column names for training
    
    # Save as parquet file - one for 'reports' and one for 'regulations'
    for domain in ["reports", "regulations"]:
        domain_data = export_data[export_data["domain"] == domain]
        if not domain_data.empty:
            output_file = os.path.join(args.export_dir, f"{domain}.parquet")
            domain_data.to_parquet(output_file)
            logger.info(f"Exported {len(domain_data)} {domain} records to {output_file}")

def main():
    args = parse_args()
    
    # Initialize Argilla and load data
    dataset = init_argilla(args)
    df = load_records(dataset, args)
    
    # Check if we should export data for retraining
    if args.export:
        export_for_retraining(df, args)
        logger.info(f"Exported annotated data to {args.export_dir}")
        if not df.empty:
            logger.info("To retrain with the exported data, run:")
            logger.info(f"python src/train.py --train_data {args.export_dir}")
        else:
            logger.warning("No data found to export. Make corrections in Argilla first.")
        
        # If only exporting, we can exit here
        if len(df) == 0 or df["corrected"].sum() == 0:
            return
    
    # Continue with evaluation and reporting
    # Run analyses
    correction_analysis = analyze_corrections(df)
    confidence_analysis = analyze_confidence(df)
    classification_metrics = generate_classification_metrics(df)
    
    # Generate report
    analyses = {
        "correction_analysis": correction_analysis,
        "confidence_analysis": confidence_analysis,
        "classification_metrics": classification_metrics
    }
    
    generate_report(analyses, args)

if __name__ == "__main__":
    main() 