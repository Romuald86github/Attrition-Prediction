variable "aws_region" {
  description = "AWS region to create resources in"
  default     = "us-east-1"
}

variable "ec2_key_pair_name" {
  description = "Name of the EC2 key pair to use for the Elastic Beanstalk environment"
}

variable "vpc_id" {
  description = "ID of the VPC to use for the Elastic Beanstalk environment"
}

variable "subnet_ids" {
  description = "List of subnet IDs to use for the Elastic Beanstalk environment"
  type        = list(string)
}

variable "docker_image_url" {
  description = "URL of the Docker image for the Flask application"
}