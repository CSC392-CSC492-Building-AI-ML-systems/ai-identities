'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

const loginSchema = z.object({
  username: z.string().min(2, "Username must be at least 2 characters"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});
  

type LoginFormData = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter(); // Initialize router here
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    mode: "onChange",          // validate while typing (or use "onBlur")
    });


  const onSubmit = async (data: LoginFormData) => {
  setLoading(true);
  setError(null);

  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const body = await res.json();

    if (!res.ok) throw new Error(body.message || "Login failed");

    setSuccess(true);

    router.push('/');
    localStorage.setItem("accessToken", body.accessToken);
    localStorage.setItem("userId", body.userId);
    localStorage.setItem("refreshToken", body.refreshToken);
  } catch (err: any) {
    setError(err.message);
  } finally {
    setLoading(false);        // ← set back to false
  }
};

  if (success) return <p>Login Succesful!</p>;

  return (
    <div className="min-h-screen flex items-center justify-center">
        <form onSubmit={handleSubmit(onSubmit)} className="max-w-md w-full mt-10 p-8 rounded-xl shadow-lg" style={{ background: '#2D2A5A' }}>
        <h2 className="text-2xl font-bold mb-6" style={{ color: '#F3F3FF' }}>Log In</h2>

        {/* USERNAME */}
        <label className="block mb-2 font-semibold" htmlFor="username" style={{ color: '#F3F3FF' }}>Username</label>
        <input
          id="username"
          type="text"
          {...register("username")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.username && (
          <p className="text-sm mb-4" style={{ color: '#FF6B6B' }}>
            {errors.username.message}
          </p>
        )}

        {/* PASSWORD */}
        <label className="block mb-2 font-semibold" htmlFor="password" style={{ color: '#F3F3FF' }}>Password</label>
        <input
        id="password"
        type="password"
        {...register("password")}
        className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.password && (
        <p className="text-sm mb-4" style={{ color: '#FF6B6B' }}>
            {errors.password.message}
        </p>
        )}
        <div className="mt-4 mb-4">
          <span style={{ color: '#9290C3' }}>New user?</span>{' '}
          <Link href="/signup" className="text-[#9290C3] underline hover:text-[#F3F3FF] transition-colors">
            Signup
          </Link>
        </div>
        <button
            type="submit"
            className="w-full py-2 rounded font-semibold transition-colors"
            style={{ background: '#9290C3', color: '#F3F3FF' }}
            onMouseEnter={e => (e.currentTarget.style.background = '#535C91')}
            onFocus={e => (e.currentTarget.style.background = '#535C91')}
            onMouseLeave={e => (e.currentTarget.style.background = '#9290C3')}
            onBlur={e => (e.currentTarget.style.background = '#9290C3')}
        >
            Log In
        </button>
        {loading ? (
          <div style={{ color: '#6d6badff' }}>Loading...</div>
        ) : (<span></span>)}
        {error && <p className="mt-4 text-center text-sm" style={{ color: '#FF6B6B' }}>{error}</p>}
        </form>
    </div>
  );
}