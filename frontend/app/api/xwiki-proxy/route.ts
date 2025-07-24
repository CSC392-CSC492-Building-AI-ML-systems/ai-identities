import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const path = req.nextUrl.searchParams.get('path');
  if (!path) {
    return NextResponse.json({ error: 'Missing path parameter' }, { status: 400 });
  }

  try {
    const cookieHeader = req.headers.get('cookie') || '';
    const xwikiRes = await fetch(`http://159.203.20.200:8080/${path}`, {
      headers: { Cookie: cookieHeader },
      credentials: 'include',
    });

    const text = await xwikiRes.text();
    return new NextResponse(text, {
      status: xwikiRes.status,
      headers: { 'Content-Type': 'text/plain' },
    });
  } catch (err) {
    console.error('Proxy error:', err);
    return NextResponse.json({ error: 'Proxy fetch failed' }, { status: 500 });
  }
}
