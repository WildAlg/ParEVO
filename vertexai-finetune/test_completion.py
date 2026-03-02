import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig


PROJECT_ID = "XXXXX"
END_POINT_ID = "XXXXX"
vertexai.init(project=PROJECT_ID, location="us-central1")

# Use your tuned model endpoint
model = GenerativeModel(f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{END_POINT_ID}")
# Or use the tuned model resource name from the job:
# model = GenerativeModel(tuning_job.tuned_model_endpoint_name)

config = GenerationConfig(
    temperature=0.7,
    max_output_tokens=2048,
)

response = model.generate_content(
    "Briefly explain what parlay lib is for?",
    generation_config=config
)

print(response.text)