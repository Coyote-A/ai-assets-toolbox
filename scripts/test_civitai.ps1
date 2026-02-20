$urls = @(
    "https://civitai.com/models/929497?modelVersionId=2247497",
    "https://civitai.com/models/100435?modelVersionId=1096293", 
    "https://civitai.com/models/1231943?modelVersionId=1736373"
)

$headers = @{
    "User-Agent" = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

$results = @()

foreach ($url in $urls) {
    try {
        $response = Invoke-WebRequest -Uri $url -Headers $headers -UseBasicParsing
        $content = $response.Content
        
        $nameMatch = [regex]::Match($content, '<meta\s+property="og:title"\s+content="([^"]+)"')
        $name = if ($nameMatch.Success) { $nameMatch.Groups[1].Value.Trim() } else { "Unknown" }
        
        $tagMatches = [regex]::Matches($content, '<a\s+href="/tags/[^"]+"\s+class="[^"]*tag[^"]*">([^<]+)</a>')
        $tags = @()
        foreach ($match in $tagMatches) {
            $tag = $match.Groups[1].Value.Trim()
            if ($tag -and $tag.Length -gt 1 -and $tags -notcontains $tag) {
                $tags += $tag
            }
        }
        
        $modelIdMatch = [regex]::Match($url, '/models/(\d+)')
        $modelId = if ($modelIdMatch.Success) { $modelIdMatch.Groups[1].Value } else { "Unknown" }
        
        $versionIdMatch = [regex]::Match($url, 'modelVersionId=(\d+)')
        $versionId = if ($versionIdMatch.Success) { $versionIdMatch.Groups[1].Value } else { "Unknown" }
        
        $result = [PSCustomObject]@{
            ModelId     = $modelId
            VersionId   = $versionId
            FullName    = $name
            Tags        = $tags
            URL         = $url
        }
        
        $results += $result
        
    }
    catch {
        $result = [PSCustomObject]@{
            ModelId     = "Unknown"
            VersionId   = "Unknown"
            FullName    = "Error"
            Tags        = @()
            URL         = $url
        }
        $results += $result
    }
}

$results | ConvertTo-Json -Depth 3 | Out-File -FilePath "civitai_results.json" -Encoding utf8
