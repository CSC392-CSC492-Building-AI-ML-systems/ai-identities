"use client";

import Link from "next/link";
import Pill from "@/components/pill";
import { useEffect, useState } from "react";
import { useRouter } from 'next/navigation';
import { usePathname } from 'next/navigation'; // Import for path change detection
import Image from "next/image";


export default function Navbar() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const router = useRouter(); 
  const pathname = usePathname();  // To track the current route

  const checkAuth = () => {
    const token = localStorage.getItem("accessToken");
    setIsAuthenticated(!!token); // Convert to boolean
  };

  const fetchNotificationCount = async () => {
    if (!isAuthenticated) return; // Only fetch if user is logged in

    try {
      const token = localStorage.getItem("accessToken");
      const userId = localStorage.getItem("userId");

      // Ensure userId is a valid string
      const headers: HeadersInit = {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      };

      if (userId) {
        headers["userid"] = userId;  // Add userId only if it's a valid string
      }
    } catch (error) {
      console.error("Failed to fetch notification count:", error);
    }
  };

  useEffect(() => {
    // Check auth status on component mount
    checkAuth();
  }, []);

  useEffect(() => {
    // Re-check auth status on route change
    checkAuth();
    fetchNotificationCount();
  }, [pathname]); // Re-run when pathname changes

  // Poll for notifications every 30 seconds
  useEffect(() => {
    if (isAuthenticated) {
      const interval = setInterval(fetchNotificationCount, 30000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated]);

  const handleAuthClick = () => {
    if (isAuthenticated) {
      // Handle logout
      localStorage.removeItem("accessToken");
      localStorage.removeItem("refreshToken");
      localStorage.removeItem("userId");
      setIsAuthenticated(false); // Update the state
      router.push("/signup"); // Redirect to login
    } else {
      // Redirect to login page if not authenticated
      router.push("/login");
    }
  };

  return (
    <nav className="bg-[#2D2A5A] shadow-md px-10 py-6 flex items-center justify-between border-b border-[#9290C3]">
      <div className="text-3xl font-normal flex items-center gap-2" style={{ color: '#F3F3FF' }}>
        <Image src="/llmdetlogo.png" alt="Logo" width={48} height={48} />
        <Link href="/">
          <span className="font-bold" style={{ color: '#F3F3FF' }}>LLM</span><span className="font-normal" style={{ color: '#F3F3FF' }}>Detective</span>
        </Link>
      </div>
      <div className="flex gap-32">
        <div className="flex gap-14">
          <Link href="/wiki" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
            Wiki
            <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          <Link href="/identify" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
            Identify
            <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
          <Link href="/about" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal px-2">
            About
            <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
          </Link>
        </div>
        <button
          onClick={handleAuthClick}
          className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal bg-transparent border-none outline-none cursor-pointer px-2"
          style={{ background: 'none', boxShadow: 'none' }}
        >
          Login
          <span className="pointer-events-none absolute left-0 bottom-[-9px] h-[0.2rem] rounded-full bg-[#9290C3] transition-all duration-300 max-w-0 group-hover:max-w-full w-full"></span>
        </button>
      </div>
    </nav>
  );
}
