"use client";

import Link from "next/link";
import { usePathname } from 'next/navigation'; // Import for path change detection
import Image from "next/image";
import { useXWikiAuth } from '../hooks/useXWikiAuth';


export default function Navbar() {
  const { loggedIn, username, loading } = useXWikiAuth();
  const redirectPath = usePathname();  // To track the current route

  console.log(loggedIn, "     ", username, "    ", loading)

  const handleLogin = () => {
    const redirectUrl = encodeURIComponent(`https://wiki.llm.test/bin/view/redir?next=${redirectPath}`);
    window.location.href = `https://wiki.llm.test/bin/login/XWiki/XWikiLogin?xredirect=${redirectUrl}`;
  };

  const handleLogout = () => {
    const redirectUrl = encodeURIComponent(`https://wiki.llm.test/bin/view/redir?next=${redirectPath}`);
    window.location.href = `https://wiki.llm.test/bin/logout/XWiki/XWikiLogout?xredirect=${redirectUrl}`;
  };

  // handle the node that shows auth status to user
  let authNode: React.ReactNode;
  if (loading) {
    authNode = (
      <span className="text-[#F3F3FF] px-2 whitespace-nowrap">Checking...</span>
    );
  } else if (loggedIn) {
    authNode = (
      <button
        onClick={handleLogout}
        className="relative group text-[#F3F3FF] text-lg font-normal px-2 flex items-center whitespace-nowrap"
      >
        Log Out
        <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
      </button>
    );
  } else {
    authNode = (
      <button
        onClick={handleLogin}
        className="relative group text-[#F3F3FF] text-lg font-normal px-2 flex items-center whitespace-nowrap"
      >
        Login
        <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
      </button>
    );
  }


  return (
    <div className="fixed top-0 left-0 w-full z-50 flex justify-center">
      <nav className="h-20 w-full max-w-5xl bg-[#2D2A5A] flex items-center justify-between px-8 rounded-2xl" style={{ boxShadow: '0 6px 24px 0 rgba(0,0,0,0.5)' }}>
        <div className="flex items-center gap-3 text-3xl font-normal" style={{ color: '#F3F3FF' }}>
          <Image src="/llmdetlogo.png" alt="Logo" width={48} height={48} />
          <Link href="/" className="flex items-center gap-1">
            <span className="font-bold" style={{ color: '#F3F3FF' }}>LLM</span><span className="font-normal" style={{ color: '#F3F3FF' }}>Detective</span>
          </Link>
        </div>
        <div className="flex items-center gap-10 justify-end">
          <Link href="/identify" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2 flex items-center">
            Identification{/*
            */}<span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          <Link href="/wiki" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2 flex items-center">
            Wiki{/*
            */}<span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          <Link href="/search" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2 flex items-center">
            Search{/*
            */}<span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          
          {authNode}
        </div>
      </nav>
    </div>
  );
}
