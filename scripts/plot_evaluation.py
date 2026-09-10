import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from src.config import SETTINGS
out=Path('evaluation/results'); gold=pd.read_csv(SETTINGS.golden_dir/'golden_set.csv'); metrics=json.loads((out/'metrics.json').read_text()); esc=json.loads((out/'escalation_metrics.json').read_text())
gold.gold_intent.value_counts().plot.bar(title='Golden intent distribution');plt.tight_layout();plt.savefig(out/'intent_distribution.png');plt.close()
pd.DataFrame({k:v['macro_f1'] for k,v in metrics.items()},index=['Macro F1']).T.plot.bar(legend=False,title='Intent macro-F1');plt.tight_layout();plt.savefig(out/'macro_f1_comparison.png');plt.close()
ConfusionMatrixDisplay(confusion_matrix=metrics['ai_agent']['confusion_matrix'],display_labels=metrics['ai_agent']['labels']).plot(xticks_rotation=45);plt.title('AI agent intent confusion matrix');plt.tight_layout();plt.savefig(out/'ai_confusion_matrix.png');plt.close()
ConfusionMatrixDisplay(confusion_matrix=esc['confusion_matrix'],display_labels=esc['labels']).plot();plt.title('Escalation confusion matrix');plt.tight_layout();plt.savefig(out/'escalation_confusion_matrix.png');plt.close()
