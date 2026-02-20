<#
.SYNOPSIS
Extracts CivitAI model details (name, UI name, tags) using web scraping with PowerShell.

.DESCRIPTION
Fetches CivitAI model pages using Invoke-WebRequest and extracts key information
using regular expressions, bypassing API authorization requirements.
#>

# CivitAI model URLs to scrape
$modelUrls = @(
    "https://civitai.com/models/929497?modelVersionId=2247497",
    "https://civitai.com/models/100435?modelVersionId=1096293", 
    "https://civitai.com/models/1231943?modelVersionId=1736373"
)

# Results array to store extracted information
$results = @()

# User-Agent header to mimic a browser
$headers = @{
    "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

foreach ($url in $modelUrls) {
    Write-Host "Fetching: $url" -ForegroundColor Cyan
    
    try {
        # Fetch the web page
        $response = Invoke-WebRequest -Uri $url -Headers $headers -UseBasicParsing
        
        # Extract model details using regex
        $content = $response.Content
        
        # Extract model name (full name)
        $nameMatch = [regex]::Match($content, '<meta\s+property="og:title"\s+content="([^"]+)"')
        $name = if ($nameMatch.Success) { $nameMatch.Groups[1].Value.Trim() } else { "Unknown" }
        
        # Extract description or tags (looking for keywords in the page)
        $tags = @()
        
        # Look for tag elements (common patterns on CivitAI)
        $tagMatches = [regex]::Matches($content, '<a\s+href="/tags/[^"]+"\s+class="[^"]*tag[^"]*">([^<]+)</a>')
        foreach ($match in $tagMatches) {
            $tag = $match.Groups[1].Value.Trim()
            if ($tag -and $tag.Length -gt 1 -and $tags -notcontains $tag) {
                $tags += $tag
            }
        }
        
        # Extract UI-friendly name (2-3 words, removing version numbers and special characters)
        $uiName = $name
        # Remove version numbers like v1.0, 1.5, etc.
        $uiName = [regex]::Replace($uiName, '\s*v?\d+(\.\d+)*\s*', ' ')
        # Remove special characters
        $uiName = [regex]::Replace($uiName, '[^\w\s]', '')
        # Split into words and take first 3
        $uiWords = $uiName.Split(' ', [System.StringSplitOptions]::RemoveEmptyEntries)
        if ($uiWords.Count -gt 3) {
            $uiName = $uiWords[0..2] -join ' '
        } else {
            $uiName = $uiWords -join ' '
        }
        
        # Extract model ID and version ID from URL
        $modelIdMatch = [regex]::Match($url, '/models/(\d+)')
        $modelId = if ($modelIdMatch.Success) { $modelIdMatch.Groups[1].Value } else { "Unknown" }
        
        $versionIdMatch = [regex]::Match($url, 'modelVersionId=(\d+)')
        $versionId = if ($versionIdMatch.Success) { $versionIdMatch.Groups[1].Value } else { "Unknown" }
        
        # Create result object
        $result = [PSCustomObject]@{
            ModelId     = $modelId
            VersionId   = $versionId
            FullName    = $name
            UIName      = $uiName
            Tags        = $tags
            URL         = $url
        }
        
        $results += $result
        
        Write-Host "Successfully extracted: $name" -ForegroundColor Green
    }
    catch {
        Write-Host "Error fetching $url : $_" -ForegroundColor Red
    }
}

# Display results
Write-Host "`n=== Extraction Results ===" -ForegroundColor Yellow
foreach ($result in $results) {
    Write-Host "`nModel ID: $($result.ModelId)" -ForegroundColor Cyan
    Write-Host "Version ID: $($result.VersionId)"
    Write-Host "Full Name: $($result.FullName)"
    Write-Host "UI Name: $($result.UIName)"
    Write-Host "Tags: $($result.Tags -join ', ')"
    Write-Host "URL: $($result.URL)"
}

# Save results to JSON file
$outputFile = "civitai_model_details.json"
$results | ConvertTo-Json -Depth 3 | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "`nResults saved to: $outputFile" -ForegroundColor Green
