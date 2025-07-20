import { NextRequest, NextResponse } from 'next/server';
import * as cookie from 'cookie';

export async function POST(req: NextRequest) {
  const { username, password } = await req.json();

  const formBody = new URLSearchParams();
  formBody.append('j_username', username);
  formBody.append('j_password', password);
  formBody.append('xredirect', '/bin/view/Main/');

  try {
    const res = await fetch('http://159.203.20.200:8080/bin/login/XWiki/XWikiLogin', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formBody.toString(),
      redirect: 'manual',
    });

    const setCookieHeader = res.headers.get('set-cookie');
    const jsessionMatch = setCookieHeader?.match(/JSESSIONID=([^;]+);/);

    if (!jsessionMatch) {
      return NextResponse.json({ message: 'Invalid credentials' }, { status: 401 });
    }

    const response = NextResponse.json({ message: 'Login successful', userId: username });
    response.headers.set(
      'Set-Cookie',
      cookie.serialize('JSESSIONID', jsessionMatch[1], {
        path: '/',
        httpOnly: true,
        secure: false, // set to true in production with HTTPS
      })
    );

    return response;
  } catch (error: any) {
    return NextResponse.json({ message: error.message }, { status: 500 });
  }
}
