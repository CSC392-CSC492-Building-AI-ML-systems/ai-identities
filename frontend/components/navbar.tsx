"use client";

import Link from "next/link";
import { usePathname } from "next/navigation"; // Import for path change detection
import Image from "next/image";
import { useXWikiAuth } from "../hooks/useXWikiAuth";
import ArrowDropDownIcon from "@mui/icons-material/ArrowDropDown";
import ArrowDropUpIcon from "@mui/icons-material/ArrowDropUp";
import * as React from "react";
import Button from "@mui/material/Button";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";

function WikiMenu() {
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <div>
      <Button
        id="wiki-button"
        aria-controls={open ? "wiki-menu" : undefined}
        aria-haspopup="true"
        aria-expanded={open ? "true" : undefined}
        onClick={handleClick}
        sx={{ fontSize: "", textTransform: "capitalize " }}
      >
        <span className="relative group text-[#F3F3FF] text-lg font-normal">
          Wiki {!open ? <ArrowDropDownIcon /> : <ArrowDropUpIcon />}
        </span>
      </Button>
      <Menu
        id="wiki-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        disableScrollLock={true}
        slotProps={{
          list: {
            "aria-labelledby": "wiki-button",
            style: {
              backgroundColor: "#2D2A5A",
              color: "white",
            },
          },
          paper: {
            style: {
              backgroundColor: "#2D2A5A",
            },
          },
        }}
      >
        <MenuItem onClick={handleClose}>
          <Link href="/search">View</Link>
        </MenuItem>
        <MenuItem onClick={handleClose}>
          <Link href="/create">Create</Link>
        </MenuItem>
      </Menu>
    </div>
  );
}

export default function Navbar() {
  const { loggedIn, username, loading } = useXWikiAuth();
  const redirectPath = usePathname(); // To track the current route

  console.log(loggedIn, "     ", username, "    ", loading);
  const redirectUrl = encodeURIComponent(
    `https://wiki.llm.test/bin/view/redir?next=${redirectPath}`
  );
  const href_logout_url = `https://wiki.llm.test/bin/logout/XWiki/XWikiLogout?xredirect=${redirectUrl}`;
  const href_login_url = `https://wiki.llm.test/bin/login/XWiki/XWikiLogin?xredirect=${redirectUrl}`;
  const href_signup_url = `https://wiki.llm.test/bin/register/XWiki/XWikiRegister`;
  return (
    <div className="fixed top-0 left-0 w-full z-50 flex justify-center px-4">
      <nav
        className="h-16 w-full max-w-7xl bg-[#2D2A5A] flex items-center justify-between px-8 rounded-2xl mt-2"
        style={{ boxShadow: "0 6px 24px 0 rgba(0,0,0,0.5)" }}
      >
        {/* Left section - Logo and main navigation */}
        <div className="flex items-center gap-8">
          <div
            className="text-2xl font-normal flex items-center gap-2"
            style={{ color: "#F3F3FF" }}
          >
            <Image src="/llmdetlogo.png" alt="Logo" width={40} height={40} />
            <Link href="/">
              <span className="font-bold" style={{ color: "#F3F3FF" }}>
                LLM
              </span>
              <span className="font-normal" style={{ color: "#F3F3FF" }}>
                Detective
              </span>
            </Link>
          </div>

          {!loading && (
            <div className="flex items-center gap-8">
              <Link
                href="/identify"
                className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 lg:text-lg font-normal"
              >
                Identification
              </Link>
              <WikiMenu />
            </div>
          )}
        </div>

        {/* Right section - Auth buttons */}
        <div>
          {loading ? (
            <span className="text-[#F3F3FF] text-sm font-normal">
              Verifying...
            </span>
          ) : (
            <div className="flex items-center gap-8">
              {loggedIn ? (
                <Link
                  href={href_logout_url}
                  className="text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal"
                >
                  Logout
                </Link>
              ) : (
                <>
                  <Link
                    href={href_login_url}
                    className="text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal"
                  >
                    Sign in
                  </Link>
                  <Link
                    href={href_signup_url}
                    className="text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 text-lg font-normal"
                  >
                    Sign up
                  </Link>
                </>
              )}
            </div>
          )}
        </div>
      </nav>
    </div>
  );
}
