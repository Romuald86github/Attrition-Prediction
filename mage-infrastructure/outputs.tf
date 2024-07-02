output "mage_db_endpoint" {
  description = "Endpoint for the Mage database"
  value       = aws_db_instance.mage_db.endpoint
}

output "mage_artifacts_bucket" {
  description = "Name of the S3 bucket for Mage artifacts"
  value       = aws_s3_bucket.mage_artifacts.id
}

output "mage_server_service_name" {
  description = "Name of the ECS service for the Mage server"
  value       = aws_ecs_service.mage_server_service.name
}