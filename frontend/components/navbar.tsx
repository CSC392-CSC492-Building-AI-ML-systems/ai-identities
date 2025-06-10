"use client";

import Link from "next/link";
import Pill from "@/components/pill";
import { useEffect, useState } from "react";
import { useRouter } from 'next/navigation';
import { usePathname } from 'next/navigation'; // Import for path change detection


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
    <nav className="bg-[#3A3E83] shadow-md px-4 py-6 flex items-center justify-between">
      <div className="text-3xl text-white ml-8">
        <Link href="/">
          <span className="font-bold">LLM</span>Detective
        </Link>
      </div>
      <div className="flex gap-16">
        <div className="flex gap-6">
          <Link href="/">
            <Pill text="Wiki" />
          </Link>
          <Link href="/">
            <Pill text="Identify" />
          </Link>
          <Link href="/">
            <Pill text="About" />
          </Link>
        </div>
        <button
          onClick={handleAuthClick}
          className="inline-block bg-white text-black px-8 py-2 rounded-full text-sm select-none"
        >
          Login
        </button>
      </div>
    </nav>
  );
}
