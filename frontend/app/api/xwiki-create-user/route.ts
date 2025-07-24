import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { username, password, email, firstName, lastName } = await req.json();

    console.log('[XWiki Signup] Received data:', { username, email, firstName, lastName });

    if (!username || !password) {
      console.warn('[XWiki Signup] Missing username or password.');
      return NextResponse.json(
        { error: 'Username and password are required' },
        { status: 400 }
      );
    }

    const xml = `
      <page>
        <title>${username}</title>
        <syntax>xwiki/2.1</syntax>
        <content></content>
        <objects>
          <object>
            <className>XWiki.XWikiUsers</className>
            <properties>
              <property><name>email</name><value>${email || ''}</value></property>
              <property><name>first_name</name><value>${firstName || ''}</value></property>
              <property><name>last_name</name><value>${lastName || ''}</value></property>
              <property><name>password</name><value>${password}</value></property>
            </properties>
          </object>
        </objects>
      </page>
    `;

    const auth = Buffer.from(
      `${process.env.XWIKI_ADMIN_USERNAME}:${process.env.XWIKI_ADMIN_PASSWORD}`
    ).toString('base64');

    console.log('[XWiki Signup] Sending request to XWiki REST API...');
    console.log('[XWiki Signup] Target URL:', `http://159.203.20.200:8080/xwiki/rest/wikis/xwiki/spaces/XWiki/pages/${username}`);
    console.log('[XWiki Signup] Auth user:', process.env.XWIKI_ADMIN_USERNAME);
    console.log('[XWiki Signup] XML Payload:', xml);

    const res = await fetch(
      `http://159.203.20.200:8080/xwiki/rest/wikis/xwiki/spaces/XWiki/pages/${username}`,
      {
        method: 'PUT',
        headers: {
          'Authorization': `Basic ${auth}`,
          'Content-Type': 'application/xml',
        },
        body: xml,
      }
    );

    const responseText = await res.text();
    console.log('[XWiki Signup] Response status:', res.status, res.statusText);
    console.log('[XWiki Signup] Response body:', responseText);

    if (!res.ok) {
      return NextResponse.json(
        { error: `XWiki API failed: ${res.status} ${res.statusText}`, details: responseText },
        { status: res.status }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('[XWiki Signup] Unexpected server error:', error);
    return NextResponse.json(
      { error: 'Unexpected server error', details: String(error) },
      { status: 500 }
    );
  }
}
