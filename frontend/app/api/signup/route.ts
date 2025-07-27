import { NextResponse } from "next/server";

const BASE_URL = "http://159.203.20.200:8080";
const WIKI = "xwiki";
const SPACE = "XWiki";
const AUTH = "Basic " + Buffer.from("ahmed33033:ahmed2003").toString("base64");

export async function POST(req: Request) {
  try {
    const { username, email, password, firstName, lastName } = await req.json();

    if (!username || !password) {
      return NextResponse.json({ error: "Username and password are required" }, { status: 400 });
    }

    // 1. Create page
    const pageUrl = `${BASE_URL}/rest/wikis/${WIKI}/spaces/${SPACE}/pages/${username}`;
    const createPage = await fetch(pageUrl, {
      method: "PUT",
      headers: { "Content-Type": "application/xml", Authorization: AUTH },
      body: `<page xmlns="http://www.xwiki.org"><title>${username}</title><content>Created via API</content></page>`,
    });

    if (!createPage.ok) {
      return NextResponse.json({ error: `Failed to create page: ${await createPage.text()}` }, { status: 400 });
    }

    // 2. Create user object
    const objectsUrl = `${BASE_URL}/rest/wikis/${WIKI}/spaces/${SPACE}/pages/${username}/objects`;
    const createObject = await fetch(objectsUrl, {
      method: "POST",
      headers: { "Content-Type": "application/xml", Authorization: AUTH },
      body: `<object xmlns="http://www.xwiki.org"><className>XWiki.XWikiUsers</className></object>`,
    });

    if (!createObject.ok) {
      return NextResponse.json({ error: `Failed to create object: ${await createObject.text()}` }, { status: 400 });
    }

    // 3. Update properties
    const properties = [
      { name: "first_name", value: firstName || "" },
      { name: "last_name", value: lastName || "" },
      { name: "email", value: email || "" },
      { name: "password", value: password },
    ];

    for (const prop of properties) {
      const propUrl = `${BASE_URL}/rest/wikis/${WIKI}/spaces/${SPACE}/pages/${username}/objects/XWiki.XWikiUsers/0/properties/${prop.name}`;
      const res = await fetch(propUrl, {
        method: "PUT",
        headers: { "Content-Type": "text/plain", Authorization: AUTH },
        body: prop.value,
      });
      if (!res.ok) {
        return NextResponse.json({ error: `Failed to set ${prop.name}: ${await res.text()}` }, { status: 400 });
      }
    }

    return NextResponse.json({ message: "User created successfully" }, { status: 201 });
  } catch (err: any) {
    return NextResponse.json({ error: err.message || "Internal Server Error" }, { status: 500 });
  }
}