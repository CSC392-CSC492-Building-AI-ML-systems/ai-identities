import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export async function POST(req: NextRequest) {
  try {
    const { username, password, email, firstName, lastName } = await req.json();

    // Basic validation
    if (!username || !password || !email) {
      return NextResponse.json({ message: 'Missing required fields' }, { status: 400 });
    }

    // Basic auth header for XWiki admin account
    const authHeader = 'Basic ' + Buffer.from('ahmed33033:ahmed33033').toString('base64');

    const response = await axios.post(
      'http://159.203.20.200:8080/bin/view/XWiki/API/RegisterUser/WebHome?xpage=plain',
      {
        username,
        password,
        email,
        firstName,
        lastName,
      },
      {
        headers: {
          'Authorization': authHeader,
          'Content-Type': 'application/json',
          'Accept': '*/*',
          'User-Agent': 'Next.js XWiki Client',
          'Connection': 'keep-alive',
        },
        maxRedirects: 0, // Don't follow login redirects
        validateStatus: () => true, // Accept all status codes for inspection
      }
    );

    const textResponse = typeof response.data === 'string'
      ? response.data
      : JSON.stringify(response.data);

    if (!textResponse.toLowerCase().includes('created successfully')) {
      return NextResponse.json({
        message: 'User creation failed',
        response: textResponse.slice(0, 300),
      }, { status: 400 });
    }

    return NextResponse.json({
      message: 'User created successfully',
      username,
    });
  } catch (err: any) {
    console.error('Register error:', err?.response?.data || err.message);
    return NextResponse.json({
      message: err?.response?.data || 'Internal server error',
    }, { status: 500 });
  }
}
