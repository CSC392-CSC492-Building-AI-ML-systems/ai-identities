import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const XWIKI_API_URL = "http://159.203.20.200:8080/rest/wikis/xwiki/spaces/Main/pages";
  const XWIKI_USERNAME = process.env.NEXT_PUBLIC_XWIKI_USERNAME || "";
  const XWIKI_PASSWORD = process.env.NEXT_PUBLIC_XWIKI_PASSWORD || "";
  const BASIC_AUTH = "Basic " + Buffer.from(`${XWIKI_USERNAME}:${XWIKI_PASSWORD}`).toString('base64');

  console.log("API Route called");
  console.log("Username:", XWIKI_USERNAME);
  console.log("Password length:", XWIKI_PASSWORD.length);
  console.log("Target URL:", XWIKI_API_URL);

  try {
    const response = await fetch(XWIKI_API_URL, {
      headers: {
        'Accept': 'application/xml',
        'Authorization': BASIC_AUTH,
      },
    });

    console.log("XWiki response status:", response.status);
    console.log("XWiki response headers:", Object.fromEntries(response.headers.entries()));

    if (!response.ok) {
      console.error("XWiki API error:", response.status, response.statusText);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const xml = await response.text();
    console.log("XWiki XML response:", xml.substring(0, 1000) + "...");
    
    return new NextResponse(xml, {
      status: 200,
      headers: {
        'Content-Type': 'application/xml',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
  } catch (error) {
    console.error('Error fetching from XWiki:', error);
    return NextResponse.json(
      { error: 'Failed to fetch from XWiki' },
      { status: 500 }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    },
  });
} 