# main.tf

provider "google" {
  project = "rl-robot-brain"  
  region  = "australia-southeast1"
}

# 1. Define the Cloud Run Service
resource "google_cloud_run_service" "rl_robot_service" {
  name     = "rl-robot-inference-v1"
  location = "australia-southeast1"

  template {
    spec {
      containers {
        # <--- REPLACE THIS with your existing container image URL
        image = "australia-southeast1-docker.pkg.dev/rl-robot-brain/cloud-run-source-deploy/rl-robot-brain@sha256:225631c642616cc0f5439ec7934ba80c79cdf8f77e20c2c7ae05b7dff34065f5" 
        
        resources {
          limits = {
            memory = "512Mi"
            cpu    = "1"
          }
        }
        ports {
            container_port = 8080 # Default Cloud Run port
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# 2. The "No Auth" Policy (Matches your current setup)
# This resource explicitly tells GCP: "Let anyone on the internet talk to this."
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_service.rl_robot_service.name
  location = google_cloud_run_service.rl_robot_service.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 3. Output the URL so you know it worked
output "robot_url" {
  value = google_cloud_run_service.rl_robot_service.status[0].url
}