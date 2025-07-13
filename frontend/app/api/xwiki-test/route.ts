import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const XWIKI_USERNAME = process.env.NEXT_PUBLIC_XWIKI_USERNAME || "";
  const XWIKI_PASSWORD = process.env.NEXT_PUBLIC_XWIKI_PASSWORD || "";
  const BASIC_AUTH = "Basic " + Buffer.from(`${XWIKI_USERNAME}:${XWIKI_PASSWORD}`).toString('base64');

  console.log("Testing XWiki API endpoints...");

  // Test different endpoints
  const endpoints = [
    "http://159.203.20.200:8080/rest/wikis/xwiki/spaces",
    "http://159.203.20.200:8080/rest/wikis/xwiki/spaces/Main/pages",
    "http://159.203.20.200:8080/rest/wikis/xwiki/spaces/Sandbox/pages",
    "http://159.203.20.200:8080/rest/wikis/xwiki/pages",
  ];

  const results = [];

  for (const endpoint of endpoints) {
    try {
      console.log(`Testing endpoint: ${endpoint}`);
      const response = await fetch(endpoint, {
        headers: {
          'Accept': 'application/xml',
          'Authorization': BASIC_AUTH,
        },
      });

      const status = response.status;
      const xml = await response.text();
      
      results.push({
        endpoint,
        status,
        hasData: xml.length > 0,
        xmlPreview: xml.substring(0, 200) + "..."
      });

      console.log(`Endpoint ${endpoint}: Status ${status}, Length: ${xml.length}`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      results.push({
        endpoint,
        status: 'ERROR',
        error: errorMessage
      });
      console.error(`Error testing ${endpoint}:`, error);
    }
  }

  return NextResponse.json({ results });
} 