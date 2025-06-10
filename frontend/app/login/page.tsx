'use client';

import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import Navbar from "@/components/navbar";

const loginSchema = z.object({
    firstName: z.string().min(2, "First name must be at least 2 characters"),
    lastName: z.string().min(2, "Last name must be at least 2 characters"),
    email: z.string().email("Invalid email address"),
    password: z.string().min(6, "Password must be at least 6 characters"),
});
  

type LoginFormData = z.infer<typeof loginSchema>;

export default function LoginPage() {
  const router = useRouter(); // Initialize router here
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
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
    <div className="min-h-screen bg-gray-950">
        <Navbar />
        <form onSubmit={handleSubmit(onSubmit)} className="max-w-md mx-auto mt-10 p-6 bg-gray-800 rounded shadow">
        <h2 className="text-2xl font-bold text-white mb-4">Log In</h2>

        {/* EMAIL */}
        <label className="block mb-2 font-semibold text-white" htmlFor="email">Email</label>
        <input
        id="email"
        type="email"
        {...register("email")}        /* register adds ref, value & onChange */
        className="w-full mb-1 px-3 py-2 border rounded"
        />
        {errors.email && (
        <p className="text-red-600 text-sm mb-4">
            {errors.email.message}
        </p>
        )}

        {/* PASSWORD */}
        <label className="block mb-2 font-semibold text-white" htmlFor="email">Password</label>
        <input
        id="password"
        type="password"
        {...register("password")}
        className="w-full mb-1 px-3 py-2 border rounded"
        />
        {errors.password && (
        <p className="text-red-600 text-sm mb-4">
            {errors.password.message}
        </p>
        )}
        <div className="mt-4 mb-4">
          New user?{' '}
          <Link href="/signup" className="text-blue-600 underline">
            Signup
          </Link>
        </div>
        <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 transition"
        >
            Log In
        </button>
        </form>
    </div>
  );
}