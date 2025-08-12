"use client";

import Link from "next/link";
import Pill from "@/components/pill";
import { useEffect, useState } from "react";
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

  return (
    <div className="fixed top-0 left-0 w-full z-50 flex justify-center">
      <nav className="h-20 w-[60%] bg-[#2D2A5A] flex items-center justify-between px-6 rounded-2xl" style={{ boxShadow: '0 6px 24px 0 rgba(0,0,0,0.5)' }}>
      <div className="text-3xl font-normal flex items-center gap-2" style={{ color: '#F3F3FF' }}>
        <Image src="/llmdetlogo.png" alt="Logo" width={48} height={48} />
        <Link href="/">
          <span className="font-bold" style={{ color: '#F3F3FF' }}>LLM</span><span className="font-normal" style={{ color: '#F3F3FF' }}>Detective</span>
        </Link>
      </div>
      <div className="flex gap-32">
        <div className="flex gap-14">
            <Link href="/identify" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
              Identification
              <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
            </Link>
          <Link href="/wiki" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
            Wiki
            <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          <Link href="/search" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
            Search
            <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
        </div>
          {loading ? (
            <span className="text-[#F3F3FF] px-2">Checking...</span>
          ) : loggedIn ? (
            <button onClick={handleLogout} className="relative group ...">
              Log Out
              <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
            </button>
          ) : (
            <button onClick={handleLogin} className="relative group ...">
              Login
              <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
            </button>
          )}
        </div>
    </nav>
    </div>
  );
}
