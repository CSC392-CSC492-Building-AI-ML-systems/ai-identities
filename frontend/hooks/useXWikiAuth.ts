'use client';

import { useState, useEffect } from 'react';

export function useXWikiAuth() {
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null);
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    async function checkAuth() {
      console.log('[useXWikiAuth] Running checkAuth()...');

      try {
        const res = await fetch(
          `/api/xwiki-proxy?path=bin/view/XWiki/CurrentUser?xpage=plain`,
          { credentials: 'include' }
        );

        console.log('[useXWikiAuth] Response status:', res.status, res.statusText);

        const text = (await res.text()).trim();
        const cleanText = text.replace(/<[^>]+>/g, '').trim(); // strip HTML tags
        console.log('[useXWikiAuth] Raw response text:', text);
        console.log('[useXWikiAuth] Clean response text:', cleanText);

        if (!res.ok) {
          console.error('[useXWikiAuth] Request failed. Treating as logged out.');
          setLoggedIn(false);
          setUsername(null);
          return;
        }

        if (text.includes('XWikiGuest')) {
          console.log('[useXWikiAuth] Detected guest user.');
          setLoggedIn(false);
          setUsername(null);
        } else {
          console.log('[useXWikiAuth] Detected logged-in user:', text);
          setLoggedIn(true);
          setUsername(text.replace('XWiki.', '').trim());
        }
      } catch (error) {
        console.error('[useXWikiAuth] Error fetching CurrentUser:', error);
        setLoggedIn(false);
        setUsername(null);
      }
    }

    checkAuth();
  }, []);

  return { loggedIn, username, loading: loggedIn === null };
}