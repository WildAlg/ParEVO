import time

import vertexai
from vertexai.tuning import sft

PROJECT_ID = "XXXXX"

vertexai.init(project=PROJECT_ID, location="us-central1")

sft_tuning_job = sft.train(
    source_model="gemini-2.5-pro",
    train_dataset="gs://XXXXX/dataset_1_gemini.jsonl",
    epochs=1,
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)
