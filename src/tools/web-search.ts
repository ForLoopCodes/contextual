// You.com web search integration for Context+ MCP server
// Provides external web knowledge to complement local semantic search

export interface WebSearchOptions {
  query: string;
  count?: number;
  offset?: number;
  country?: string;
  domains?: string[];
  freshness?: string;
  search_lang?: string;
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  thumbnail?: {
    url: string;
  };
}

export interface WebSearchResponse {
  web?: {
    results: WebSearchResult[];
  };
  news?: {
    results: WebSearchResult[];
  };
}

/**
 * Performs web search using You.com Search API
 * Falls back to keyless operation if no API key provided
 */
export async function webSearch(options: WebSearchOptions): Promise<string> {
  const {
    query,
    count = 10,
    offset = 0,
    country = "us",
    search_lang = "en",
    domains,
    freshness
  } = options;

  // Check for API key (optional)
  const apiKey = process.env.YDC_API_KEY;
  
  try {
    // Build query parameters
    const params = new URLSearchParams({
      query,
      count: Math.min(Math.max(count, 1), 20).toString(), // Clamp between 1-20
      offset: Math.max(offset, 0).toString(),
      country,
      search_lang,
    });

    if (domains && domains.length > 0) {
      params.append('domains', domains.join(','));
    }
    if (freshness) {
      params.append('freshness', freshness);
    }

    // Make API request
    const url = `https://api.you.com/v1/agents/search?${params}`;
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': 'youdotcom-integration/forloopcodes-contextplus'
    };

    if (apiKey) {
      headers['X-API-Key'] = apiKey;
    }

    const response = await fetch(url, { 
      method: 'GET',
      headers
    });

    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Invalid API key. Check your YDC_API_KEY environment variable.');
      }
      if (response.status === 429) {
        throw new Error('Rate limit exceeded. Consider setting YDC_API_KEY for higher quotas.');
      }
      throw new Error(`Web search failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json() as WebSearchResponse;
    
    return formatSearchResults(data, query);
    
  } catch (error) {
    if (error instanceof Error) {
      return `Web search error: ${error.message}`;
    }
    return 'Web search failed due to an unexpected error.';
  }
}

/**
 * Formats search results into a readable text format
 */
function formatSearchResults(data: WebSearchResponse, query: string): string {
  const results: string[] = [`Web Search Results for: "${query}"\n`];
  
  // Process web results
  if (data.web?.results && data.web.results.length > 0) {
    results.push('🔍 Web Results:');
    data.web.results.forEach((result, index) => {
      results.push(`${index + 1}. **${result.title}**`);
      results.push(`   URL: ${result.url}`);
      if (result.snippet) {
        results.push(`   ${result.snippet.replace(/\n/g, ' ').trim()}`);
      }
      results.push('');
    });
  }

  // Process news results
  if (data.news?.results && data.news.results.length > 0) {
    results.push('📰 News Results:');
    data.news.results.forEach((result, index) => {
      results.push(`${index + 1}. **${result.title}**`);
      results.push(`   URL: ${result.url}`);
      if (result.snippet) {
        results.push(`   ${result.snippet.replace(/\n/g, ' ').trim()}`);
      }
      results.push('');
    });
  }

  if (!data.web?.results?.length && !data.news?.results?.length) {
    results.push('No web search results found. Try a different query.');
  }

  return results.join('\n');
}