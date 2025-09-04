// useXWikiAuth.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import { XWIKI_URL } from "@/constants";

type Options = {
  xwikiSrc?: string;
  iframeId?: string;
  timeoutMs?: number;
};

export function useXWikiAuth({
  xwikiSrc = `${XWIKI_URL}/bin/view/xwiki_auth_page`,
  iframeId = "xwiki-auth-bridge",
  timeoutMs = 5000,
}: Options = {}) {
  const [loggedIn, setLoggedIn] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Always compute origin from the src to avoid mismatches
  const xwikiOrigin = useMemo(() => new URL(xwikiSrc).origin, [xwikiSrc]);

  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (e.origin !== xwikiOrigin) return; // trust only XWiki
      const d = e.data;
      if (d && d.type === "xwiki-auth") {
        setLoggedIn(Boolean(d.loggedIn));
        setUsername(d.username ?? null);
        setLoading(false);
      }
    }

    window.addEventListener("message", onMessage);

    // Ensure a single hidden iframe exists (create if missing)
    let iframe = document.getElementById(iframeId) as HTMLIFrameElement | null;
    if (!iframe) {
      iframe = document.createElement("iframe");
      iframe.id = iframeId;
      iframe.src = xwikiSrc;
      Object.assign(iframe.style, {
        position: "absolute",
        width: "0",
        height: "0",
        border: "0",
        visibility: "hidden",
      } as Partial<CSSStyleDeclaration>);
      iframe.setAttribute("aria-hidden", "true");
      document.body.appendChild(iframe);
    }

    // Ping AFTER the iframe has navigated to XWiki (prevents the mismatch error)
    const onLoad = () => {
      iframe?.contentWindow?.postMessage({ type: "xwiki-auth-request" }, xwikiOrigin);
    };
    iframe.addEventListener("load", onLoad, { once: true });

    // Fallback: if the iframe was already loaded before this hook mounted,
    // this won't fire 'load', so send a permissive ping that won't error.
    iframe.contentWindow?.postMessage({ type: "xwiki-auth-request" }, xwikiOrigin);

    const t = window.setTimeout(() => setLoading(false), timeoutMs);

    return () => {
      window.removeEventListener("message", onMessage);
      iframe?.removeEventListener("load", onLoad);
      window.clearTimeout(t);
      // leave the iframe in DOM for reuse
    };
  }, [xwikiSrc, xwikiOrigin, iframeId, timeoutMs]);

  return { loggedIn, username, loading };
}
