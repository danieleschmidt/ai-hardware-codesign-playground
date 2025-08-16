# CloudFront CDN Configuration for Global Distribution
# AI Hardware Co-Design Playground - Global Content Delivery

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

# Local variables for CDN configuration
locals {
  # Origin configurations
  origins = {
    api = {
      domain_name = var.api_domain
      origin_id   = "api-origin"
      path        = "/api"
    }
    frontend = {
      domain_name = var.frontend_domain
      origin_id   = "frontend-origin"
      path        = "/"
    }
    static = {
      domain_name = var.static_assets_domain
      origin_id   = "static-origin"
      path        = "/static"
    }
  }
  
  # Cache behaviors for different content types
  cache_behaviors = {
    api = {
      path_pattern           = "/api/*"
      target_origin_id       = local.origins.api.origin_id
      viewer_protocol_policy = "redirect-to-https"
      allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
      cached_methods         = ["GET", "HEAD", "OPTIONS"]
      ttl_default           = 0     # No caching for API
      ttl_max              = 86400  # 1 day max
      ttl_min              = 0
      compress             = true
    }
    static = {
      path_pattern           = "/static/*"
      target_origin_id       = local.origins.static.origin_id
      viewer_protocol_policy = "redirect-to-https"
      allowed_methods        = ["GET", "HEAD"]
      cached_methods         = ["GET", "HEAD"]
      ttl_default           = 86400   # 1 day
      ttl_max              = 31536000 # 1 year
      ttl_min              = 0
      compress             = true
    }
    assets = {
      path_pattern           = "/assets/*"
      target_origin_id       = local.origins.static.origin_id
      viewer_protocol_policy = "redirect-to-https"
      allowed_methods        = ["GET", "HEAD"]
      cached_methods         = ["GET", "HEAD"]
      ttl_default           = 2592000  # 30 days
      ttl_max              = 31536000  # 1 year
      ttl_min              = 86400     # 1 day
      compress             = true
    }
  }
  
  # Geographic restrictions
  geo_restrictions = var.enable_geo_restrictions ? {
    restriction_type = "whitelist"
    locations        = var.allowed_countries
  } : {
    restriction_type = "none"
    locations        = []
  }
  
  common_tags = {
    Project     = "AI Hardware Co-Design Playground"
    Environment = var.environment
    Component   = "CDN"
    ManagedBy   = "Terraform"
  }
}

# WAF Web ACL for CloudFront
resource "aws_wafv2_web_acl" "cloudfront_waf" {
  name  = "${var.project_name}-${var.environment}-cloudfront-waf"
  scope = "CLOUDFRONT"

  default_action {
    allow {}
  }

  # Rate limiting rule
  rule {
    name     = "RateLimitRule"
    priority = 1

    override_action {
      none {}
    }

    statement {
      rate_based_statement {
        limit              = var.rate_limit_per_5min
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }

    action {
      block {}
    }
  }

  # Geo blocking rule
  dynamic "rule" {
    for_each = var.enable_geo_restrictions ? [1] : []
    content {
      name     = "GeoBlockingRule"
      priority = 2

      override_action {
        none {}
      }

      statement {
        geo_match_statement {
          country_codes = var.blocked_countries
        }
      }

      visibility_config {
        cloudwatch_metrics_enabled = true
        metric_name                = "GeoBlockingRule"
        sampled_requests_enabled   = true
      }

      action {
        block {}
      }
    }
  }

  # SQL injection protection
  rule {
    name     = "SQLInjectionRule"
    priority = 3

    override_action {
      none {}
    }

    statement {
      sqli_match_statement {
        field_to_match {
          body {}
        }
        text_transformation {
          priority = 1
          type     = "URL_DECODE"
        }
        text_transformation {
          priority = 2
          type     = "HTML_ENTITY_DECODE"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "SQLInjectionRule"
      sampled_requests_enabled   = true
    }

    action {
      block {}
    }
  }

  # XSS protection
  rule {
    name     = "XSSRule"
    priority = 4

    override_action {
      none {}
    }

    statement {
      xss_match_statement {
        field_to_match {
          body {}
        }
        text_transformation {
          priority = 1
          type     = "URL_DECODE"
        }
        text_transformation {
          priority = 2
          type     = "HTML_ENTITY_DECODE"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "XSSRule"
      sampled_requests_enabled   = true
    }

    action {
      block {}
    }
  }

  tags = local.common_tags

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-${var.environment}-cloudfront-waf"
    sampled_requests_enabled   = true
  }
}

# CloudFront Origin Access Control
resource "aws_cloudfront_origin_access_control" "main" {
  name                              = "${var.project_name}-${var.environment}-oac"
  description                       = "Origin Access Control for ${var.project_name}"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

# CloudFront Distribution
resource "aws_cloudfront_distribution" "main" {
  # Origins configuration
  dynamic "origin" {
    for_each = local.origins
    content {
      domain_name = origin.value.domain_name
      origin_id   = origin.value.origin_id
      origin_path = origin.value.path

      custom_origin_config {
        http_port              = 80
        https_port             = 443
        origin_protocol_policy = "https-only"
        origin_ssl_protocols   = ["TLSv1.2"]
      }

      origin_shield {
        enabled              = var.enable_origin_shield
        origin_shield_region = var.origin_shield_region
      }
    }
  }

  # Default cache behavior
  default_cache_behavior {
    allowed_methods            = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods             = ["GET", "HEAD"]
    target_origin_id           = local.origins.frontend.origin_id
    compress                   = true
    viewer_protocol_policy     = "redirect-to-https"
    cache_policy_id           = aws_cloudfront_cache_policy.frontend.id
    origin_request_policy_id  = aws_cloudfront_origin_request_policy.frontend.id
    response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id

    # Lambda@Edge functions
    dynamic "lambda_function_association" {
      for_each = var.enable_lambda_edge ? [1] : []
      content {
        event_type   = "viewer-request"
        lambda_arn   = aws_lambda_function.edge_security[0].qualified_arn
        include_body = false
      }
    }

    # CloudFront Functions
    dynamic "function_association" {
      for_each = var.enable_cloudfront_functions ? [1] : []
      content {
        event_type   = "viewer-request"
        function_arn = aws_cloudfront_function.url_rewrite[0].arn
      }
    }
  }

  # Ordered cache behaviors
  dynamic "ordered_cache_behavior" {
    for_each = local.cache_behaviors
    content {
      path_pattern             = ordered_cache_behavior.value.path_pattern
      allowed_methods          = ordered_cache_behavior.value.allowed_methods
      cached_methods          = ordered_cache_behavior.value.cached_methods
      target_origin_id        = ordered_cache_behavior.value.target_origin_id
      compress                = ordered_cache_behavior.value.compress
      viewer_protocol_policy  = ordered_cache_behavior.value.viewer_protocol_policy

      min_ttl     = ordered_cache_behavior.value.ttl_min
      default_ttl = ordered_cache_behavior.value.ttl_default
      max_ttl     = ordered_cache_behavior.value.ttl_max

      cache_policy_id = ordered_cache_behavior.key == "api" ? 
        aws_cloudfront_cache_policy.api.id : 
        aws_cloudfront_cache_policy.static.id

      origin_request_policy_id = ordered_cache_behavior.key == "api" ?
        aws_cloudfront_origin_request_policy.api.id :
        aws_cloudfront_origin_request_policy.static.id

      response_headers_policy_id = aws_cloudfront_response_headers_policy.security_headers.id
    }
  }

  # Aliases
  aliases = var.cloudfront_aliases

  # Price class
  price_class = var.cloudfront_price_class

  # Geographic restrictions
  restrictions {
    geo_restriction {
      restriction_type = local.geo_restrictions.restriction_type
      locations        = local.geo_restrictions.locations
    }
  }

  # SSL certificate
  viewer_certificate {
    acm_certificate_arn      = var.ssl_certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  # Custom error responses
  dynamic "custom_error_response" {
    for_each = var.custom_error_responses
    content {
      error_code            = custom_error_response.value.error_code
      response_code         = custom_error_response.value.response_code
      response_page_path    = custom_error_response.value.response_page_path
      error_caching_min_ttl = custom_error_response.value.error_caching_min_ttl
    }
  }

  # Logging
  dynamic "logging_config" {
    for_each = var.enable_access_logging ? [1] : []
    content {
      include_cookies = false
      bucket         = aws_s3_bucket.cloudfront_logs[0].bucket_domain_name
      prefix         = "cloudfront-logs/"
    }
  }

  # WAF
  web_acl_id = aws_wafv2_web_acl.cloudfront_waf.arn

  # Additional configuration
  enabled             = true
  is_ipv6_enabled    = true
  comment            = "CloudFront distribution for ${var.project_name} ${var.environment}"
  default_root_object = "index.html"
  wait_for_deployment = false

  tags = local.common_tags
}

# Cache Policies
resource "aws_cloudfront_cache_policy" "frontend" {
  name        = "${var.project_name}-${var.environment}-frontend-cache"
  comment     = "Cache policy for frontend content"
  default_ttl = 86400  # 1 day
  max_ttl     = 31536000  # 1 year
  min_ttl     = 1

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    headers_config {
      header_behavior = "whitelist"
      headers {
        items = ["CloudFront-Viewer-Country", "CloudFront-Is-Mobile-Viewer"]
      }
    }

    query_strings_config {
      query_string_behavior = "whitelist"
      query_strings {
        items = ["version", "locale"]
      }
    }

    cookies_config {
      cookie_behavior = "none"
    }
  }
}

resource "aws_cloudfront_cache_policy" "api" {
  name        = "${var.project_name}-${var.environment}-api-cache"
  comment     = "Cache policy for API endpoints"
  default_ttl = 0      # No caching by default
  max_ttl     = 86400  # 1 day max
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    headers_config {
      header_behavior = "whitelist"
      headers {
        items = [
          "Authorization",
          "Content-Type",
          "User-Agent",
          "X-Forwarded-For",
          "CloudFront-Viewer-Country"
        ]
      }
    }

    query_strings_config {
      query_string_behavior = "all"
    }

    cookies_config {
      cookie_behavior = "all"
    }
  }
}

resource "aws_cloudfront_cache_policy" "static" {
  name        = "${var.project_name}-${var.environment}-static-cache"
  comment     = "Cache policy for static assets"
  default_ttl = 86400    # 1 day
  max_ttl     = 31536000 # 1 year
  min_ttl     = 86400    # 1 day

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    headers_config {
      header_behavior = "none"
    }

    query_strings_config {
      query_string_behavior = "whitelist"
      query_strings {
        items = ["v", "version"]
      }
    }

    cookies_config {
      cookie_behavior = "none"
    }
  }
}

# Origin Request Policies
resource "aws_cloudfront_origin_request_policy" "frontend" {
  name    = "${var.project_name}-${var.environment}-frontend-origin"
  comment = "Origin request policy for frontend"

  headers_config {
    header_behavior = "whitelist"
    headers {
      items = [
        "CloudFront-Viewer-Country",
        "CloudFront-Is-Mobile-Viewer",
        "User-Agent"
      ]
    }
  }

  query_strings_config {
    query_string_behavior = "all"
  }

  cookies_config {
    cookie_behavior = "whitelist"
    cookies {
      items = ["session-id", "csrf-token"]
    }
  }
}

resource "aws_cloudfront_origin_request_policy" "api" {
  name    = "${var.project_name}-${var.environment}-api-origin"
  comment = "Origin request policy for API"

  headers_config {
    header_behavior = "allViewer"
  }

  query_strings_config {
    query_string_behavior = "all"
  }

  cookies_config {
    cookie_behavior = "all"
  }
}

resource "aws_cloudfront_origin_request_policy" "static" {
  name    = "${var.project_name}-${var.environment}-static-origin"
  comment = "Origin request policy for static assets"

  headers_config {
    header_behavior = "whitelist"
    headers {
      items = ["CloudFront-Viewer-Country"]
    }
  }

  query_strings_config {
    query_string_behavior = "whitelist"
    query_strings {
      items = ["v", "version"]
    }
  }

  cookies_config {
    cookie_behavior = "none"
  }
}

# Response Headers Policy
resource "aws_cloudfront_response_headers_policy" "security_headers" {
  name    = "${var.project_name}-${var.environment}-security-headers"
  comment = "Security headers policy"

  security_headers_config {
    strict_transport_security {
      access_control_max_age_sec = 31536000  # 1 year
      include_subdomains         = true
      preload                   = true
    }

    content_type_options {
      override = true
    }

    frame_options {
      frame_option = "DENY"
      override     = true
    }

    referrer_policy {
      referrer_policy = "strict-origin-when-cross-origin"
      override        = true
    }
  }

  custom_headers_config {
    items {
      header   = "X-Custom-Header"
      value    = "${var.project_name}-${var.environment}"
      override = false
    }

    items {
      header   = "Cache-Control"
      value    = "public, max-age=31536000"
      override = false
    }
  }

  cors_config {
    access_control_allow_credentials = false
    access_control_max_age_sec      = 86400

    access_control_allow_headers {
      items = ["*"]
    }

    access_control_allow_methods {
      items = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]
    }

    access_control_allow_origins {
      items = var.cors_allowed_origins
    }

    access_control_expose_headers {
      items = ["X-Custom-Header"]
    }

    origin_override = false
  }
}

# CloudFront Function for URL rewriting
resource "aws_cloudfront_function" "url_rewrite" {
  count   = var.enable_cloudfront_functions ? 1 : 0
  name    = "${var.project_name}-${var.environment}-url-rewrite"
  runtime = "cloudfront-js-1.0"
  comment = "URL rewrite function"
  publish = true

  code = <<-EOT
function handler(event) {
    var request = event.request;
    var uri = request.uri;
    
    // Add index.html to requests that don't have a file extension
    if (uri.endsWith('/')) {
        request.uri += 'index.html';
    } else if (!uri.includes('.')) {
        request.uri += '/index.html';
    }
    
    return request;
}
EOT
}

# Lambda@Edge function for enhanced security
resource "aws_lambda_function" "edge_security" {
  count            = var.enable_lambda_edge ? 1 : 0
  filename         = "edge-security.zip"
  function_name    = "${var.project_name}-${var.environment}-edge-security"
  role            = aws_iam_role.lambda_edge[0].arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.edge_security[0].output_base64sha256
  runtime         = "nodejs18.x"
  publish         = true

  tags = local.common_tags
}

# Lambda@Edge source code
data "archive_file" "edge_security" {
  count       = var.enable_lambda_edge ? 1 : 0
  type        = "zip"
  output_path = "edge-security.zip"
  
  source {
    content = <<-EOT
exports.handler = async (event) => {
    const request = event.Records[0].cf.request;
    const headers = request.headers;
    
    // Add security headers
    const securityHeaders = {
        'x-frame-options': [{ key: 'X-Frame-Options', value: 'DENY' }],
        'x-content-type-options': [{ key: 'X-Content-Type-Options', value: 'nosniff' }],
        'x-xss-protection': [{ key: 'X-XSS-Protection', value: '1; mode=block' }],
        'strict-transport-security': [{
            key: 'Strict-Transport-Security',
            value: 'max-age=31536000; includeSubDomains; preload'
        }]
    };
    
    // Block requests from certain user agents
    const userAgent = headers['user-agent'] ? headers['user-agent'][0].value : '';
    const blockedAgents = ['malicious-bot', 'scanner'];
    
    for (const agent of blockedAgents) {
        if (userAgent.toLowerCase().includes(agent)) {
            return {
                status: '403',
                statusDescription: 'Forbidden',
                body: 'Access Denied'
            };
        }
    }
    
    Object.assign(headers, securityHeaders);
    return request;
};
EOT
    filename = "index.js"
  }
}

# IAM role for Lambda@Edge
resource "aws_iam_role" "lambda_edge" {
  count = var.enable_lambda_edge ? 1 : 0
  name  = "${var.project_name}-${var.environment}-lambda-edge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = [
            "lambda.amazonaws.com",
            "edgelambda.amazonaws.com"
          ]
        }
      }
    ]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_edge_execution" {
  count      = var.enable_lambda_edge ? 1 : 0
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
  role       = aws_iam_role.lambda_edge[0].name
}

# S3 bucket for CloudFront access logs
resource "aws_s3_bucket" "cloudfront_logs" {
  count  = var.enable_access_logging ? 1 : 0
  bucket = "${var.project_name}-${var.environment}-cloudfront-logs-${random_id.bucket_suffix.hex}"

  tags = local.common_tags
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudfront_logs" {
  count  = var.enable_access_logging ? 1 : 0
  bucket = aws_s3_bucket.cloudfront_logs[0].id

  rule {
    id     = "log_lifecycle"
    status = "Enabled"

    expiration {
      days = var.log_retention_days
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# CloudWatch alarms for CloudFront
resource "aws_cloudwatch_metric_alarm" "cloudfront_4xx_errors" {
  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-4xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "4xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "5"
  alarm_description   = "This metric monitors CloudFront 4xx error rate"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    DistributionId = aws_cloudfront_distribution.main.id
  }

  tags = local.common_tags
}

resource "aws_cloudwatch_metric_alarm" "cloudfront_5xx_errors" {
  alarm_name          = "${var.project_name}-${var.environment}-cloudfront-5xx-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "5xxErrorRate"
  namespace           = "AWS/CloudFront"
  period              = "300"
  statistic           = "Average"
  threshold           = "1"
  alarm_description   = "This metric monitors CloudFront 5xx error rate"
  alarm_actions       = [var.sns_topic_arn]

  dimensions = {
    DistributionId = aws_cloudfront_distribution.main.id
  }

  tags = local.common_tags
}

# Outputs
output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain_name" {
  description = "CloudFront distribution domain name"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "cloudfront_hosted_zone_id" {
  description = "CloudFront hosted zone ID"
  value       = aws_cloudfront_distribution.main.hosted_zone_id
}

output "waf_web_acl_arn" {
  description = "WAF Web ACL ARN"
  value       = aws_wafv2_web_acl.cloudfront_waf.arn
}