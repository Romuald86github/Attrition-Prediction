variable "aws_region" {
  description = "AWS region to create resources in"
  default     = "us-east-1"
}

variable "mage_db_password" {
  description = "Password for the Mage database"
  sensitive   = true
}