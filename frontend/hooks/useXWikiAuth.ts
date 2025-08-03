import { useEffect, useState } from 'react';

export function useXWikiAuth() {
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const checkAuth = async () => {
      try {
        const res = await fetch('http://159.203.20.200:8080/bin/view/XWiki/CurrentUser', {
          method: 'GET',
          credentials: 'include',
          redirect: 'manual',
        });

        if (res.status === 200) {
          const text = await res.text();

          // Parse the HTML using DOMParser (safer than regex)
          const parser = new DOMParser();
          const doc = parser.parseFromString(text, 'text/html');

          const htmlEl = doc.querySelector('html');
          const userRef = htmlEl?.getAttribute('data-xwiki-user-reference');

          if (userRef && userRef.startsWith('xwiki:XWiki.')) {
            const username = userRef.split('xwiki:XWiki.')[1];
            setUsername(username);
            setLoggedIn(true);
          } else {
            setUsername(null);
            setLoggedIn(false);
          }
        } else {
          setUsername(null);
          setLoggedIn(false);
        }
      } catch (err) {
        console.error('Auth check failed:', err);
        setUsername(null);
        setLoggedIn(false);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  return { loggedIn, username, loading };
}
