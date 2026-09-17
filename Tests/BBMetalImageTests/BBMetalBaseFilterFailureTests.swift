import Metal
import ObjectiveC
import XCTest
@testable import BBMetalImage

final class BBMetalBaseFilterFailureTests: XCTestCase {
    func testCommandBufferFailureDoesNotLeaveFilterLocked() throws {
        try assertFilterRecovers(failingEncoder: false)
    }

    func testEncoderFailureDoesNotLeaveFilterLocked() throws {
        try assertFilterRecovers(failingEncoder: true)
    }

    private func assertFilterRecovers(failingEncoder: Bool) throws {
        let device = BBMetalDevice.sharedDevice
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: 8, height: 8, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        let texture = try XCTUnwrap(device.makeTexture(descriptor: descriptor))
        let source = BBMetalStaticImageSource(texture: texture)
        let filter = BBMetalPassThroughFilter(createTexture: true)
        filter.add(source: source)
        let queue = BBMetalDevice.sharedCommandQueue
        let buffer = try XCTUnwrap(queue.makeCommandBuffer())
        let target: AnyObject = failingEncoder ? buffer : queue
        let selector = NSSelectorFromString(failingEncoder ? "computeCommandEncoder" : "commandBuffer")
        let targetClass: AnyClass = try XCTUnwrap(object_getClass(target))
        let method = try XCTUnwrap(class_getInstanceMethod(targetClass, selector))
        let original = method_getImplementation(method)
        let failure: @convention(block) (AnyObject) -> AnyObject? = { _ in nil }
        let replacement = imp_implementationWithBlock(failure)
        // Fail Metal allocation only during this synchronous frame submission.
        method_setImplementation(method, replacement)
        filter.newTextureAvailable(BBMetalDefaultTexture(metalTexture: texture), from: source)
        method_setImplementation(method, original)
        imp_removeBlock(replacement)

        let recovered = expectation(description: "Filter releases its lock after allocation failure")
        DispatchQueue.global().async {
            _ = filter.sources
            filter.remove(source: source)
            recovered.fulfill()
        }
        wait(for: [recovered], timeout: 1)
    }
}
