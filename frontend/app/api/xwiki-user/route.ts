import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const response = await fetch(
      'http://159.203.20.200:8080/bin/view/XWiki/API/RegisterUser/WebHome?xpage=plain',
      { method: 'GET' }
    );

    const text = await response.text();

    // Determine login state based on presence of 'XWikiGuest'
    const isGuest = text.includes('XWikiGuest');

    // Return a proper JSON object
    return NextResponse.json({
      loggedIn: !isGuest,
      raw: text, // Optional: for debugging
    });
  } catch (error) {
    console.error('Error checking XWiki login:', error);
    return NextResponse.json(
      { loggedIn: false, error: 'Failed to reach XWiki' },
      { status: 500 }
    );
  }
}
