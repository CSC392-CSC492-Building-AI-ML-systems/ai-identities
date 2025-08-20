"use client";

import Link from "next/link";
import { usePathname } from 'next/navigation'; // Import for path change detection
import Image from "next/image";
import { useXWikiAuth } from '../hooks/useXWikiAuth';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import * as React from 'react';
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';

function BasicMenu() {
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
        id="basic-button"
        aria-controls={open ? 'basic-menu' : undefined}
        aria-haspopup="true"
        aria-expanded={open ? 'true' : undefined}
        onClick={handleClick}
        sx={{ fontSize: '', textTransform: 'capitalize ' }}
      >
        <span className="relative group text-[#F3F3FF] lg:text-lg font-normal px-2">Wiki {!open ? <ArrowDropDownIcon /> : <ArrowDropUpIcon />}</span>
      </Button>
      <Menu
        id="basic-menu"
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        slotProps={{
          list: {
            'aria-labelledby': 'basic-button',
            style: {
              backgroundColor: "#2D2A5A",
              color: "white",
            }
          },
          paper:{
            style:{
              backgroundColor: "#2D2A5A",
            }
          }
        }}
      >
        <MenuItem onClick={handleClose}>
          <Link href="/search" >
            View
          </Link>
        </MenuItem>
        <MenuItem onClick={handleClose}>
          <Link href="/create" >
            Create
          </Link>
        </MenuItem>
      </Menu>
    </div>
  );
}


export default function Navbar() {
  const { loggedIn, username, loading } = useXWikiAuth();
  const redirectPath = usePathname();  // To track the current route

  console.log(loggedIn, "     ", username, "    ", loading)
  const redirectUrl = encodeURIComponent(`https://wiki.llm.test/bin/view/redir?next=${redirectPath}`)
  const href_logout_url = `https://wiki.llm.test/bin/logout/XWiki/XWikiLogout?xredirect=${redirectUrl}`;
  const href_login_url = `https://wiki.llm.test/bin/login/XWiki/XWikiLogin?xredirect=${redirectUrl}`
  const href_signup_url = `https://wiki.llm.test/bin/register/XWiki/XWikiRegister`
  return (
    <div className="fixed top-0 left-0 w-full z-50 flex justify-center">
      <nav className="h-20 min-w-[60%] bg-[#2D2A5A] flex items-center flex-col lg:flex-row justify-between px-6 rounded-2xl" style={{ boxShadow: '0 6px 24px 0 rgba(0,0,0,0.5)' }}>
        <div className="text-3xl font-normal flex items-center gap-2" style={{ color: '#F3F3FF' }}>
          <Image src="/llmdetlogo.png" alt="Logo" width={48} height={48} />
          <Link href="/">
            <span className="font-bold" style={{ color: '#F3F3FF' }}>LLM</span><span className="font-normal" style={{ color: '#F3F3FF' }}>Detective</span>
          </Link>
        </div>
        <div>
          {loading ?
            (
              <span className="relative group text-[#F3F3FF] lg:text-lg font-normal px-2"> (Verifying authentication...)</span>
            )
            :
            <div style={{ display: "flex", justifyContent: "space-between", width: "100%", alignItems: "center" }} className="lg:gap-x-10">
              <Link href="/identify" className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 lg:text-lg font-normal px-2">
                Identification
              </Link>
              <BasicMenu />
              {loggedIn ? (
                <Link href={href_logout_url} className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 lg:text-lg font-normal px-2">
                  Logout
                </Link>
              ) : (
                <>
                  <Link href={href_login_url} className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 lg:text-lg font-normal px-2">
                    Login
                  </Link>
                  <Link href={href_signup_url} className="relative group text-[#F3F3FF] hover:text-[#9290C3] transition-colors duration-200 lg:text-lg font-normal px-2">
                    Signup
                  </Link>
                </>
              )}
            </div>
          }

        </div>
      </nav>
    </div>
  );
}