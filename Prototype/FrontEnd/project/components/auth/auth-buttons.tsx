"use client";

import { useState } from "react";
import { UserCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function AuthButtons() {
  const [isOpen, setIsOpen] = useState(false);

  const handleSubmitLogin = async (event:any) => {
    event.preventDefault(); 

    const email = event.target.elements["email-login"].value;
    const password = event.target.elements["password-login"].value;

    console.log("Email:", email);
    console.log("Password:", password);
  };

  const handleSubmitSignUp = async(event: any) => {
    event.preventDefault(); 

    const username = event.target.elements["email-signup"].value;
    const password = event.target.elements["password-signup"].value;
    const confirmPassword = event.target.elements["confirm-password"].value;

    const response = await fetch("/api/auth/signup", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: 'testuser',
        password: 'testpassword'
      }),
  });

    const data = await response.json();
    if (data.success) {
      console.log("Signup Successful");
    } else {
      console.error("Signup Failed:", data.error);
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" className="bg-card/80">
          <UserCircle className="h-5 w-5" />
          <span className="sr-only">Sign In</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Welcome to FeatherFind</DialogTitle>
          <DialogDescription>
            Sign in to your account or create a new one to start identifying birds.
          </DialogDescription>
        </DialogHeader>
        <Tabs defaultValue="login" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="login">Login</TabsTrigger>
            <TabsTrigger value="signup">Sign Up</TabsTrigger>
          </TabsList>
          <TabsContent value="login">
            <form onSubmit={handleSubmitLogin} className="space-y-4 pt-4">
              <div className="space-y-2">
                <Label htmlFor="email-login">Email</Label>
                <Input id="email-login" type="email" required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password-login">Password</Label>
                <Input id="password-login" type="password" required />
              </div>
              <Button type="submit" className="w-full">Login</Button>
            </form>
          </TabsContent>
          <TabsContent value="signup">
            <form onSubmit={handleSubmitSignUp} className="space-y-4 pt-4">
              <div className="space-y-2">
                <Label htmlFor="email-signup">Email</Label>
                <Input id="email-signup" type="email" required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password-signup">Password</Label>
                <Input id="password-signup" type="password" required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="confirm-password">Confirm Password</Label>
                <Input id="confirm-password" type="password" required />
              </div>
              <Button type="submit" className="w-full">Sign Up</Button>
            </form>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  );
}