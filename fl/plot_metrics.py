import os
import matplotlib.pyplot as plt
from fl.db import SessionLocal, RoundMetric
import sys

# Ensure shared and root are in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def plot_metrics():
    session = SessionLocal()
    try:
        metrics = session.query(RoundMetric).order_by(RoundMetric.round_number).all()
        
        if not metrics:
            print("No metrics found in the database. Run the simulation first.")
            return

        rounds: list[int] = [m.round_number for m in metrics]  # type: ignore
        accuracies: list[float] = [m.accuracy for m in metrics]  # type: ignore
        losses: list[float] = [m.loss for m in metrics]  # type: ignore

        plt.figure(figsize=(10, 5))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b')
        plt.title('Global Model Validation Accuracy')
        plt.xlabel('Round Number')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(rounds)

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(rounds, losses, marker='o', linestyle='-', color='r')
        plt.title('Global Model Validation Loss')
        plt.xlabel('Round Number')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.xticks(rounds)

        plt.tight_layout()
        
        output_dir = "docs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, 'convergence_plot.png')
        plt.savefig(output_path)
        print(f"Convergence plot saved to {output_path}")

    finally:
        session.close()

if __name__ == "__main__":
    plot_metrics()
