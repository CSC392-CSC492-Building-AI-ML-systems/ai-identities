"use client";

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Navbar from "@/components/navbar";

const signupSchema = z.object({
  username: z.string().min(3, "Username must be at least 3 characters"),
  email: z.string().email("Invalid email address").optional(),
  password: z.string().min(6, "Password must be at least 6 characters"),
  firstName: z.string().optional(),
  lastName: z.string().optional(),
});

type SignupFormData = z.infer<typeof signupSchema>;

export default function SignupPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SignupFormData>({
    resolver: zodResolver(signupSchema),
    mode: "onChange",
  });

  const onSubmit = async (data: SignupFormData) => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/signup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const body = await res.json();

      if (!res.ok) {
        throw new Error(body.error || "Signup failed");
      }

      setSuccess(true);
      // Redirect after 2 seconds to login
      setTimeout(() => router.push('/'), 2000);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (success) return <p className="text-center text-white mt-10">Signup successful! Redirecting...</p>;

  return (
    <div className="min-h-screen flex items-center justify-center bg-black">
      <Navbar />
      <form onSubmit={handleSubmit(onSubmit)} className="w-full max-w-lg p-8 rounded-xl shadow-lg bg-[#2D2A5A]">
        <h2 className="text-2xl font-bold text-[#F3F3FF] mb-6">Sign Up</h2>

        {/* First Name */}
        <label htmlFor="firstName" className="block mb-2 font-semibold text-[#F3F3FF]">First Name</label>
        <input
          id="firstName"
          type="text"
          {...register("firstName")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.username && <p className="text-sm mb-4 text-[#FF6B6B]">{errors.username.message}</p>}

        {/* Last Name */}
        <label htmlFor="lastName" className="block mb-2 font-semibold text-[#F3F3FF]">Last Name</label>
        <input
          id="lastName"
          type="text"
          {...register("lastName")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.username && <p className="text-sm mb-4 text-[#FF6B6B]">{errors.username.message}</p>}

        {/* Username */}
        <label htmlFor="username" className="block mb-2 font-semibold text-[#F3F3FF]">Username</label>
        <input
          id="username"
          type="text"
          {...register("username")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.username && <p className="text-sm mb-4 text-[#FF6B6B]">{errors.username.message}</p>}

        {/* Email */}
        <label htmlFor="email" className="block mb-2 font-semibold text-[#F3F3FF]">Email</label>
        <input
          id="email"
          type="email"
          {...register("email")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.email && <p className="text-sm mb-4 text-[#FF6B6B]">{errors.email.message}</p>}

        {/* Password */}
        <label htmlFor="password" className="block mb-2 font-semibold text-[#F3F3FF]">Password</label>
        <input
          id="password"
          type="password"
          {...register("password")}
          className="w-full mb-1 px-3 py-2 rounded bg-[#393E7C] text-[#F3F3FF] border border-[#9290C3] focus:outline-none focus:ring-2 focus:ring-[#9290C3]"
        />
        {errors.password && <p className="text-sm mb-4 text-[#FF6B6B]">{errors.password.message}</p>}

        <button
          type="submit"
          className="w-full mt-4 py-2 rounded font-semibold transition-colors"
          style={{ background: '#9290C3', color: '#F3F3FF' }}
          onMouseEnter={e => (e.currentTarget.style.background = '#535C91')}
          onFocus={e => (e.currentTarget.style.background = '#535C91')}
          onMouseLeave={e => (e.currentTarget.style.background = '#9290C3')}
          onBlur={e => (e.currentTarget.style.background = '#9290C3')}
          disabled={loading}
        >
          {loading ? "Signing up..." : "Sign Up"}
        </button>

        {error && <p className="mt-4 text-center text-sm text-[#FF6B6B]">{error}</p>}
      </form>
    </div>
  );
}
