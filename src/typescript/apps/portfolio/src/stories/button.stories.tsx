import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ArrowUpRight } from "lucide-react";

import { ButtonLink } from "@/components/ui/button";

const meta = {
  title: "UI/Button link",
  component: ButtonLink,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <div style={{ padding: "3rem" }}>
        <Story />
      </div>
    ),
  ],
  args: {
    href: "#",
    children: "Open paper",
  },
} satisfies Meta<typeof ButtonLink>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Solid: Story = {};
export const Outline: Story = { args: { variant: "outline" } };
export const WithIcon: Story = {
  render: (args) => (
    <ButtonLink {...args}>
      Open paper <ArrowUpRight aria-hidden="true" size={16} />
    </ButtonLink>
  ),
};
